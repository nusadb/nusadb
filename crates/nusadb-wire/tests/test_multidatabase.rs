//! Multiple databases over the wire (physical model, DB2/DB3/DB4/DB6): a connection is routed to its
//! startup database's engine, `CREATE`/`DROP DATABASE` act on the cluster, databases are isolated,
//! and an unknown database is refused at startup.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    reason = "integration test: unwrap/panic-on-failure is the assertion mechanism"
)]
#![allow(
    clippy::significant_drop_tightening,
    reason = "the mock cluster holds its lock across the read-then-mutate of one operation, mirroring \
              the real DatabaseManager's atomic catalog updates"
)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_wire::{
    BackendMessage, ClusterError, Connection, DatabaseCluster, FrontendMessage, PROTOCOL_VERSION,
    ServerConfig, serve_cluster_with_shutdown,
};
use tokio::net::{TcpListener, TcpStream};

/// An in-memory multiple-database cluster mirroring `DatabaseManager`'s semantics (one engine per
/// database, isolation, the same create/drop guards) without touching the filesystem.
struct MockCluster {
    engines: Mutex<HashMap<String, Arc<dyn StorageEngine>>>,
}

impl MockCluster {
    fn new() -> Self {
        let mut engines: HashMap<String, Arc<dyn StorageEngine>> = HashMap::new();
        engines.insert("nusadb".to_owned(), Arc::new(BtreeEngine::new()));
        Self {
            engines: Mutex::new(engines),
        }
    }
}

impl DatabaseCluster for MockCluster {
    fn open(&self, name: &str) -> Result<Option<Arc<dyn StorageEngine>>, ClusterError> {
        Ok(self.engines.lock().unwrap().get(name).cloned())
    }

    fn create(&self, name: &str, if_not_exists: bool) -> Result<bool, ClusterError> {
        let mut engines = self.engines.lock().unwrap();
        if engines.contains_key(name) {
            return if if_not_exists {
                Ok(false)
            } else {
                Err(ClusterError::AlreadyExists(name.to_owned()))
            };
        }
        engines.insert(name.to_owned(), Arc::new(BtreeEngine::new()));
        Ok(true)
    }

    fn drop_database(
        &self,
        name: &str,
        if_exists: bool,
        connected: &str,
    ) -> Result<bool, ClusterError> {
        if name == connected {
            return Err(ClusterError::InUse(name.to_owned()));
        }
        if name == "nusadb" {
            return Err(ClusterError::Unsupported(
                "cannot drop the default database".to_owned(),
            ));
        }
        let mut engines = self.engines.lock().unwrap();
        if engines.remove(name).is_some() {
            Ok(true)
        } else if if_exists {
            Ok(false)
        } else {
            Err(ClusterError::NotFound(name.to_owned()))
        }
    }

    fn list(&self) -> Vec<String> {
        let mut names: Vec<String> = self.engines.lock().unwrap().keys().cloned().collect();
        names.sort();
        names
    }
}

/// The outcome of a simple query, distilled from the backend message stream.
#[derive(Debug, PartialEq, Eq)]
enum Outcome {
    /// A non-row statement completed with this command tag.
    Done(String),
    /// `n` data rows were returned.
    Rows(usize),
    /// The statement errored with this SQLSTATE.
    Error(String),
}

/// Open a connection and complete the startup handshake for `database`. Returns the connection, or
/// the `(code, _)` of the fatal error if the database does not exist.
async fn connect(
    addr: std::net::SocketAddr,
    database: &str,
) -> Result<Connection<TcpStream>, String> {
    let mut client = Connection::new(TcpStream::connect(addr).await.unwrap());
    client
        .write_frame(
            &FrontendMessage::Startup {
                major: PROTOCOL_VERSION.0,
                minor: PROTOCOL_VERSION.1,
                user: "u".to_owned(),
                database: database.to_owned(),
            }
            .encode()
            .unwrap(),
        )
        .await
        .unwrap();
    loop {
        let Some(frame) = client.read_frame().await.unwrap() else {
            return Err("closed".to_owned());
        };
        match BackendMessage::decode(&frame).unwrap() {
            BackendMessage::ReadyForQuery(_) => return Ok(client),
            BackendMessage::Error { code, .. } => return Err(code),
            _ => {},
        }
    }
}

/// Run a simple query and distil its outcome (rows / command tag / error).
async fn run(client: &mut Connection<TcpStream>, sql: &str) -> Outcome {
    client
        .write_frame(
            &FrontendMessage::Query {
                sql: sql.to_owned(),
            }
            .encode()
            .unwrap(),
        )
        .await
        .unwrap();
    let mut rows = 0;
    let mut tag = None;
    let mut err = None;
    loop {
        let frame = client
            .read_frame()
            .await
            .unwrap()
            .expect("server closed mid-query");
        match BackendMessage::decode(&frame).unwrap() {
            BackendMessage::DataRow { .. } => rows += 1,
            BackendMessage::CommandComplete { tag: t } => tag = Some(t),
            BackendMessage::Error { code, .. } => err = Some(code),
            BackendMessage::ReadyForQuery(_) => break,
            _ => {},
        }
    }
    err.map_or_else(
        || {
            if rows > 0 {
                Outcome::Rows(rows)
            } else {
                Outcome::Done(tag.unwrap_or_default())
            }
        },
        Outcome::Error,
    )
}

/// Run a query and return its first row's first column as text (for `SELECT current_database()`).
async fn scalar(client: &mut Connection<TcpStream>, sql: &str) -> String {
    client
        .write_frame(
            &FrontendMessage::Query {
                sql: sql.to_owned(),
            }
            .encode()
            .unwrap(),
        )
        .await
        .unwrap();
    let mut value = None;
    loop {
        let frame = client
            .read_frame()
            .await
            .unwrap()
            .expect("server closed mid-query");
        match BackendMessage::decode(&frame).unwrap() {
            BackendMessage::DataRow { values } if value.is_none() => {
                value = values
                    .into_iter()
                    .next()
                    .flatten()
                    .map(|b| String::from_utf8_lossy(&b).into_owned());
            },
            BackendMessage::ReadyForQuery(_) => break,
            _ => {},
        }
    }
    value.expect("a scalar value")
}

/// Run a query and return the first column of every row, in order (for the `nusadb_databases`
/// catalog listing).
async fn column_values(client: &mut Connection<TcpStream>, sql: &str) -> Vec<String> {
    client
        .write_frame(
            &FrontendMessage::Query {
                sql: sql.to_owned(),
            }
            .encode()
            .unwrap(),
        )
        .await
        .unwrap();
    let mut out = Vec::new();
    loop {
        let frame = client
            .read_frame()
            .await
            .unwrap()
            .expect("server closed mid-query");
        match BackendMessage::decode(&frame).unwrap() {
            BackendMessage::DataRow { values } => out.push(
                values
                    .into_iter()
                    .next()
                    .flatten()
                    .map(|b| String::from_utf8_lossy(&b).into_owned())
                    .unwrap_or_default(),
            ),
            BackendMessage::ReadyForQuery(_) => break,
            _ => {},
        }
    }
    out
}

#[tokio::test]
async fn multiple_databases_create_isolate_and_drop_over_the_wire() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cluster: Arc<dyn DatabaseCluster> = Arc::new(MockCluster::new());
    let server = tokio::spawn(serve_cluster_with_shutdown(
        listener,
        Arc::clone(&cluster),
        ServerConfig::default(),
        std::future::pending::<()>(),
    ));

    // Connecting to the bootstrap default works; CREATE DATABASE registers a new one.
    let mut def = connect(addr, "nusadb").await.expect("connect default");
    assert_eq!(
        run(&mut def, "CREATE DATABASE shop").await,
        Outcome::Done("CREATE DATABASE".to_owned())
    );
    assert!(cluster.list().contains(&"shop".to_owned()));
    // A duplicate without IF NOT EXISTS is the duplicate-database code; with it, a no-op.
    assert_eq!(
        run(&mut def, "CREATE DATABASE shop").await,
        Outcome::Error("42P04".to_owned())
    );
    assert_eq!(
        run(&mut def, "CREATE DATABASE IF NOT EXISTS shop").await,
        Outcome::Done("CREATE DATABASE".to_owned())
    );

    // A second connection to `shop` writes a table there.
    let mut shop = connect(addr, "shop").await.expect("connect shop");
    assert_eq!(
        run(&mut shop, "CREATE TABLE t (id INT NOT NULL)").await,
        Outcome::Done("CREATE TABLE".to_owned())
    );
    assert_eq!(
        run(&mut shop, "INSERT INTO t VALUES (7)").await,
        Outcome::Done("INSERT 1".to_owned())
    );
    assert_eq!(run(&mut shop, "SELECT id FROM t").await, Outcome::Rows(1));

    // The default database is isolated: it has no `t` (separate engine).
    assert!(matches!(
        run(&mut def, "SELECT id FROM t").await,
        Outcome::Error(_)
    ));

    // CURRENT_DATABASE() names the database each connection is bound to.
    assert_eq!(scalar(&mut shop, "SELECT current_database()").await, "shop");
    assert_eq!(
        scalar(&mut def, "SELECT current_database()").await,
        "nusadb"
    );

    // Connecting to a database that does not exist is refused at startup (3D000).
    assert_eq!(connect(addr, "ghost").await.unwrap_err(), "3D000");

    // DROP DATABASE removes it; dropping the database the connection is in is refused (55006).
    assert_eq!(
        run(&mut def, "DROP DATABASE nusadb").await,
        Outcome::Error("55006".to_owned())
    );
    // `shop` still has a live connection, but the in-memory cluster does not track that, so the drop
    // succeeds here — the point under test is the wire routing of the command, not the refcount guard
    // (covered by the DatabaseManager unit tests).
    drop(shop);
    assert_eq!(
        run(&mut def, "DROP DATABASE shop").await,
        Outcome::Done("DROP DATABASE".to_owned())
    );
    assert!(!cluster.list().contains(&"shop".to_owned()));

    server.abort();
}

#[tokio::test]
async fn nusadb_databases_catalog_lists_the_cluster() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cluster: Arc<dyn DatabaseCluster> = Arc::new(MockCluster::new());
    let server = tokio::spawn(serve_cluster_with_shutdown(
        listener,
        Arc::clone(&cluster),
        ServerConfig::default(),
        std::future::pending::<()>(),
    ));

    let mut def = connect(addr, "nusadb").await.expect("connect default");
    // `SHOW` reads only a configuration parameter, so `SHOW DATABASES` — a non-standard listing
    // statement, not a parameter — is rejected loudly (unrecognized-parameter error), while
    // `SHOW <config-param>` still works.
    assert!(
        matches!(run(&mut def, "SHOW DATABASES").await, Outcome::Error(code) if code == "42704"),
        "SHOW DATABASES must be rejected as an unknown parameter",
    );
    assert!(
        matches!(run(&mut def, "show databases").await, Outcome::Error(_)),
        "the rejection is case-insensitive",
    );
    // A real config parameter is untouched (`SHOW` still reads GUCs).
    assert!(matches!(
        run(&mut def, "SHOW search_path").await,
        Outcome::Rows(_)
    ));

    // The supported replacement: `SELECT name FROM nusadb_databases` lists the cluster via the
    // engine's own `nusadb_*` system catalog. Initially only the bootstrap default.
    assert_eq!(
        column_values(&mut def, "SELECT name FROM nusadb_databases").await,
        vec!["nusadb".to_owned()]
    );

    // Newly created databases appear; the relation supports real SQL — projection, ORDER BY, WHERE.
    run(&mut def, "CREATE DATABASE shop").await;
    run(&mut def, "CREATE DATABASE analytics").await;
    assert_eq!(
        column_values(&mut def, "SELECT name FROM nusadb_databases ORDER BY name").await,
        vec![
            "analytics".to_owned(),
            "nusadb".to_owned(),
            "shop".to_owned(),
        ]
    );
    assert_eq!(
        column_values(
            &mut def,
            "SELECT * FROM nusadb_databases WHERE name = 'shop'"
        )
        .await,
        vec!["shop".to_owned()]
    );

    server.abort();
}

#[tokio::test]
async fn database_ops_are_refused_inside_a_transaction_block() {
    // CREATE/DROP DATABASE are non-transactional: they must be refused inside an open transaction
    // block — both an active one and an aborted-but-still-open (`Failed`) one — never silently run
    // (an irreversible physical drop from a doomed transaction would be data loss).
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cluster: Arc<dyn DatabaseCluster> = Arc::new(MockCluster::new());
    let server = tokio::spawn(serve_cluster_with_shutdown(
        listener,
        Arc::clone(&cluster),
        ServerConfig::default(),
        std::future::pending::<()>(),
    ));

    let mut c = connect(addr, "nusadb").await.expect("connect");
    assert_eq!(
        run(&mut c, "CREATE DATABASE keepme").await,
        Outcome::Done("CREATE DATABASE".to_owned())
    );

    // Active transaction block: DROP DATABASE is refused (25001) and `keepme` survives.
    assert_eq!(
        run(&mut c, "BEGIN").await,
        Outcome::Done("BEGIN".to_owned())
    );
    assert_eq!(
        run(&mut c, "DROP DATABASE keepme").await,
        Outcome::Error("25001".to_owned())
    );
    assert_eq!(
        run(&mut c, "ROLLBACK").await,
        Outcome::Done("ROLLBACK".to_owned())
    );
    assert!(cluster.list().contains(&"keepme".to_owned()));

    // Aborted (Failed) transaction block: a prior error doomed the transaction, but DROP DATABASE is
    // still refused — not executed (the bug this guards against missed the `Failed` state).
    assert_eq!(
        run(&mut c, "BEGIN").await,
        Outcome::Done("BEGIN".to_owned())
    );
    assert!(matches!(
        run(&mut c, "SELECT * FROM no_such_table").await,
        Outcome::Error(_)
    ));
    assert_eq!(
        run(&mut c, "DROP DATABASE keepme").await,
        Outcome::Error("25001".to_owned())
    );
    assert!(
        cluster.list().contains(&"keepme".to_owned()),
        "the database must survive a drop attempted from an aborted transaction"
    );
    assert_eq!(
        run(&mut c, "ROLLBACK").await,
        Outcome::Done("ROLLBACK".to_owned())
    );

    // In auto-commit the same DROP succeeds.
    assert_eq!(
        run(&mut c, "DROP DATABASE keepme").await,
        Outcome::Done("DROP DATABASE".to_owned())
    );
    assert!(!cluster.list().contains(&"keepme".to_owned()));

    server.abort();
}

// --- LISTEN / NOTIFY async pub/sub (phase 2) --------------------------

/// Read the next backend frame, expecting an asynchronous `NotificationResponse`. Returns
/// `(pid, channel, payload)`.
async fn next_notification(client: &mut Connection<TcpStream>) -> (u32, String, String) {
    let frame = client
        .read_frame()
        .await
        .unwrap()
        .expect("server closed before delivering a notification");
    match BackendMessage::decode(&frame).unwrap() {
        BackendMessage::NotificationResponse {
            pid,
            channel,
            payload,
        } => (pid, channel, payload),
        other => panic!("expected NotificationResponse, got {other:?}"),
    }
}

#[tokio::test]
async fn notify_delivers_to_a_listener_on_the_same_channel() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cluster: Arc<dyn DatabaseCluster> = Arc::new(MockCluster::new());
    let server = tokio::spawn(serve_cluster_with_shutdown(
        listener,
        Arc::clone(&cluster),
        ServerConfig::default(),
        std::future::pending::<()>(),
    ));

    // A listens on `orders`; a repeat LISTEN is a harmless no-op.
    let mut a = connect(addr, "nusadb").await.expect("connect a");
    assert_eq!(
        run(&mut a, "LISTEN orders").await,
        Outcome::Done("LISTEN".to_owned())
    );
    assert_eq!(
        run(&mut a, "LISTEN orders").await,
        Outcome::Done("LISTEN".to_owned())
    );

    // B notifies the channel with a payload.
    let mut b = connect(addr, "nusadb").await.expect("connect b");
    assert_eq!(
        run(&mut b, "NOTIFY orders, 'row 42'").await,
        Outcome::Done("NOTIFY".to_owned())
    );

    // A receives it asynchronously while idle.
    let (_pid, channel, payload) = next_notification(&mut a).await;
    assert_eq!(channel, "orders");
    assert_eq!(payload, "row 42");

    // A payload-less NOTIFY delivers an empty payload.
    assert_eq!(
        run(&mut b, "NOTIFY orders").await,
        Outcome::Done("NOTIFY".to_owned())
    );
    let (_pid, channel, payload) = next_notification(&mut a).await;
    assert_eq!(channel, "orders");
    assert_eq!(payload, "");

    // After UNLISTEN, A no longer receives — prove it by notifying a second channel A does listen on
    // and checking that arrives next (the un-listened one never does).
    assert_eq!(
        run(&mut a, "UNLISTEN orders").await,
        Outcome::Done("UNLISTEN".to_owned())
    );
    assert_eq!(
        run(&mut a, "LISTEN shipments").await,
        Outcome::Done("LISTEN".to_owned())
    );
    assert_eq!(
        run(&mut b, "NOTIFY orders, 'ignored'").await,
        Outcome::Done("NOTIFY".to_owned())
    );
    assert_eq!(
        run(&mut b, "NOTIFY shipments, 'seen'").await,
        Outcome::Done("NOTIFY".to_owned())
    );
    let (_pid, channel, payload) = next_notification(&mut a).await;
    assert_eq!(
        channel, "shipments",
        "the un-listened `orders` must not arrive"
    );
    assert_eq!(payload, "seen");

    server.abort();
}

#[tokio::test]
async fn notify_is_scoped_per_database_and_self_delivers() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cluster: Arc<dyn DatabaseCluster> = Arc::new(MockCluster::new());
    let server = tokio::spawn(serve_cluster_with_shutdown(
        listener,
        Arc::clone(&cluster),
        ServerConfig::default(),
        std::future::pending::<()>(),
    ));

    let mut def = connect(addr, "nusadb").await.expect("connect default");
    run(&mut def, "CREATE DATABASE shop").await;

    // A listener on `nusadb`; a notifier on `shop` sends the same channel name.
    let mut a = connect(addr, "nusadb").await.expect("connect a");
    assert_eq!(
        run(&mut a, "LISTEN news").await,
        Outcome::Done("LISTEN".to_owned())
    );
    let mut shop = connect(addr, "shop").await.expect("connect shop");
    assert_eq!(
        run(&mut shop, "NOTIFY news, 'cross-db'").await,
        Outcome::Done("NOTIFY".to_owned())
    );

    // The cross-database notification must NOT reach A. Prove it by having A self-notify on `nusadb`
    // and asserting that (the same-database one) is what arrives next — self-delivery works, and the
    // cross-database notify was correctly filtered out.
    assert_eq!(
        run(&mut a, "NOTIFY news, 'same-db'").await,
        Outcome::Done("NOTIFY".to_owned())
    );
    let (_pid, channel, payload) = next_notification(&mut a).await;
    assert_eq!(channel, "news");
    assert_eq!(
        payload, "same-db",
        "only the same-database (self) notification should arrive"
    );

    server.abort();
}

#[tokio::test]
async fn notify_in_transaction_delivers_on_commit_and_dedups() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cluster: Arc<dyn DatabaseCluster> = Arc::new(MockCluster::new());
    let server = tokio::spawn(serve_cluster_with_shutdown(
        listener,
        Arc::clone(&cluster),
        ServerConfig::default(),
        std::future::pending::<()>(),
    ));

    // `l` listens; `c` notifies inside a transaction. A per-test channel name isolates this from the
    // other tests sharing the process-global registry (same database `nusadb`).
    let mut l = connect(addr, "nusadb").await.expect("connect listener");
    assert_eq!(
        run(&mut l, "LISTEN txn_commit").await,
        Outcome::Done("LISTEN".to_owned())
    );
    let mut c = connect(addr, "nusadb").await.expect("connect actor");

    // NOTIFY inside a transaction is accepted (queued) but not yet delivered.
    assert_eq!(
        run(&mut c, "BEGIN").await,
        Outcome::Done("BEGIN".to_owned())
    );
    assert_eq!(
        run(&mut c, "NOTIFY txn_commit, 'a'").await,
        Outcome::Done("NOTIFY".to_owned())
    );
    assert_eq!(
        run(&mut c, "NOTIFY txn_commit, 'a'").await, // identical -> deduped on flush
        Outcome::Done("NOTIFY".to_owned())
    );
    assert_eq!(
        run(&mut c, "NOTIFY txn_commit, 'b'").await,
        Outcome::Done("NOTIFY".to_owned())
    );

    // COMMIT flushes the queued notifications, in order, with the duplicate 'a' collapsed.
    assert_eq!(
        run(&mut c, "COMMIT").await,
        Outcome::Done("COMMIT".to_owned())
    );
    let (_pid, channel, payload) = next_notification(&mut l).await;
    assert_eq!((channel.as_str(), payload.as_str()), ("txn_commit", "a"));
    let (_pid, channel, payload) = next_notification(&mut l).await;
    assert_eq!(
        (channel.as_str(), payload.as_str()),
        ("txn_commit", "b"),
        "the duplicate 'a' must not be delivered a second time before 'b'"
    );

    server.abort();
}

#[tokio::test]
async fn notify_in_transaction_is_discarded_on_rollback() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cluster: Arc<dyn DatabaseCluster> = Arc::new(MockCluster::new());
    let server = tokio::spawn(serve_cluster_with_shutdown(
        listener,
        Arc::clone(&cluster),
        ServerConfig::default(),
        std::future::pending::<()>(),
    ));

    let mut l = connect(addr, "nusadb").await.expect("connect listener");
    assert_eq!(
        run(&mut l, "LISTEN txn_rollback").await,
        Outcome::Done("LISTEN".to_owned())
    );
    let mut c = connect(addr, "nusadb").await.expect("connect actor");

    // A NOTIFY that is rolled back must never be delivered.
    assert_eq!(
        run(&mut c, "BEGIN").await,
        Outcome::Done("BEGIN".to_owned())
    );
    assert_eq!(
        run(&mut c, "NOTIFY txn_rollback, 'rolled-back'").await,
        Outcome::Done("NOTIFY".to_owned())
    );
    assert_eq!(
        run(&mut c, "ROLLBACK").await,
        Outcome::Done("ROLLBACK".to_owned())
    );

    // Prove the rolled-back notification is gone by sending a sentinel and asserting it arrives first
    // (per-connection delivery is FIFO, so a leaked 'rolled-back' would arrive before 'after').
    assert_eq!(
        run(&mut c, "NOTIFY txn_rollback, 'after'").await,
        Outcome::Done("NOTIFY".to_owned())
    );
    let (_pid, channel, payload) = next_notification(&mut l).await;
    assert_eq!(
        (channel.as_str(), payload.as_str()),
        ("txn_rollback", "after"),
        "the rolled-back notification must not be delivered"
    );

    server.abort();
}

#[tokio::test]
async fn notify_rollback_to_savepoint_discards_only_the_later_ones() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cluster: Arc<dyn DatabaseCluster> = Arc::new(MockCluster::new());
    let server = tokio::spawn(serve_cluster_with_shutdown(
        listener,
        Arc::clone(&cluster),
        ServerConfig::default(),
        std::future::pending::<()>(),
    ));

    let mut l = connect(addr, "nusadb").await.expect("connect listener");
    assert_eq!(
        run(&mut l, "LISTEN txn_sp").await,
        Outcome::Done("LISTEN".to_owned())
    );
    let mut c = connect(addr, "nusadb").await.expect("connect actor");

    // NOTIFY before a savepoint survives ROLLBACK TO; one after it is discarded.
    assert_eq!(
        run(&mut c, "BEGIN").await,
        Outcome::Done("BEGIN".to_owned())
    );
    assert_eq!(
        run(&mut c, "NOTIFY txn_sp, 'before-sp'").await,
        Outcome::Done("NOTIFY".to_owned())
    );
    assert_eq!(
        run(&mut c, "SAVEPOINT sp").await,
        Outcome::Done("SAVEPOINT".to_owned())
    );
    assert_eq!(
        run(&mut c, "NOTIFY txn_sp, 'after-sp'").await,
        Outcome::Done("NOTIFY".to_owned())
    );
    assert_eq!(
        run(&mut c, "ROLLBACK TO SAVEPOINT sp").await,
        Outcome::Done("ROLLBACK".to_owned())
    );
    assert_eq!(
        run(&mut c, "NOTIFY txn_sp, 'after-rollback'").await,
        Outcome::Done("NOTIFY".to_owned())
    );
    assert_eq!(
        run(&mut c, "COMMIT").await,
        Outcome::Done("COMMIT".to_owned())
    );

    // Only 'before-sp' and 'after-rollback' survive; 'after-sp' was discarded by ROLLBACK TO.
    let (_pid, _ch, payload) = next_notification(&mut l).await;
    assert_eq!(payload, "before-sp");
    let (_pid, _ch, payload) = next_notification(&mut l).await;
    assert_eq!(
        payload, "after-rollback",
        "'after-sp' must have been discarded by ROLLBACK TO SAVEPOINT"
    );

    server.abort();
}

#[tokio::test]
async fn notify_in_a_failed_transaction_is_rejected() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cluster: Arc<dyn DatabaseCluster> = Arc::new(MockCluster::new());
    let server = tokio::spawn(serve_cluster_with_shutdown(
        listener,
        Arc::clone(&cluster),
        ServerConfig::default(),
        std::future::pending::<()>(),
    ));

    let mut c = connect(addr, "nusadb").await.expect("connect");
    assert_eq!(
        run(&mut c, "BEGIN").await,
        Outcome::Done("BEGIN".to_owned())
    );
    // Force the transaction into the aborted state.
    assert!(matches!(
        run(&mut c, "SELECT * FROM does_not_exist").await,
        Outcome::Error(_)
    ));
    // NOTIFY in an aborted transaction is rejected like any other command (25P02), not queued.
    assert_eq!(
        run(&mut c, "NOTIFY txn_failed, 'x'").await,
        Outcome::Error("25P02".to_owned())
    );
    assert_eq!(
        run(&mut c, "ROLLBACK").await,
        Outcome::Done("ROLLBACK".to_owned())
    );

    server.abort();
}
