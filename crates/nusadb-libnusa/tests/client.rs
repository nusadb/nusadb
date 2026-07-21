//! Integration tests for libnusa driven against a real in-process `nusadb-server`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "integration test harness asserts via unwrap/expect"
)]

use std::sync::Arc;

use nusadb_btree::BtreeEngine;
use nusadb_core::StorageEngine;
use nusadb_libnusa::{Client, Config, Param, Pool};
use nusadb_wire::{AuthStore, ServerConfig, serve, serve_with_shutdown};
use tokio::net::TcpListener;

/// Boot a trust-on-startup server on an ephemeral port; returns its `host`/`port` and the join
/// handle (abort it to stop the server).
async fn boot_server() -> (String, u16, tokio::task::JoinHandle<std::io::Result<()>>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());
    let handle = tokio::spawn(serve(listener, engine));
    (addr.ip().to_string(), addr.port(), handle)
}

#[tokio::test]
async fn simple_query_round_trip() {
    let (host, port, server) = boot_server().await;
    let config = Config::new(host, port, "u", "nusadb");
    let mut client = Client::connect(&config).await.unwrap();

    let r = client
        .simple_query("CREATE TABLE t (id INT NOT NULL, name TEXT)")
        .await
        .unwrap();
    assert_eq!(r.command_tag(), Some("CREATE TABLE"));

    let n = client
        .execute("INSERT INTO t VALUES (5, 'alice')", &[])
        .await
        .unwrap();
    assert_eq!(n, 1);

    let result = client.simple_query("SELECT id, name FROM t").await.unwrap();
    assert_eq!(&*result.columns, ["id".to_owned(), "name".to_owned()]);
    // Protocol 1.1 typed metadata: the client negotiates minor=1, so the server reports
    // each column's type alongside its name.
    assert_eq!(
        &*result.column_types,
        [Some("INT".to_owned()), Some("TEXT".to_owned())]
    );
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].get_i64(0).unwrap(), Some(5));
    assert_eq!(
        result.rows[0].get_string(1).unwrap(),
        Some("alice".to_owned())
    );
    assert_eq!(result.affected(), Some(1)); // SELECT 1

    client.close().await.unwrap();
    server.abort();
}

#[tokio::test]
async fn extended_query_with_parameters() {
    let (host, port, server) = boot_server().await;
    let config = Config::new(host, port, "u", "nusadb");
    let mut client = Client::connect(&config).await.unwrap();

    client
        .simple_query("CREATE TABLE t (id INT NOT NULL, name TEXT)")
        .await
        .unwrap();
    // Parameterised insert through the extended-query path.
    let n = client
        .execute(
            "INSERT INTO t VALUES ($1, $2)",
            &[1_i64.into(), "alice".into()],
        )
        .await
        .unwrap();
    assert_eq!(n, 1);
    client
        .execute(
            "INSERT INTO t VALUES ($1, $2)",
            &[2_i64.into(), Param::null()],
        )
        .await
        .unwrap();

    // Parameterised SELECT.
    let result = client
        .query(
            "SELECT id, name FROM t WHERE id = $1",
            &[Param::from(2_i64)],
        )
        .await
        .unwrap();
    assert_eq!(result.rows.len(), 1);
    assert_eq!(result.rows[0].get_i64(0).unwrap(), Some(2));
    assert!(result.rows[0].is_null(1)); // name was NULL

    client.close().await.unwrap();
    server.abort();
}

#[tokio::test]
async fn prepared_statement_reused() {
    let (host, port, server) = boot_server().await;
    let config = Config::new(host, port, "u", "nusadb");
    let mut client = Client::connect(&config).await.unwrap();

    client
        .simple_query("CREATE TABLE t (id INT NOT NULL)")
        .await
        .unwrap();
    for i in 1..=3_i64 {
        client
            .execute("INSERT INTO t VALUES ($1)", &[i.into()])
            .await
            .unwrap();
    }

    let stmt = client
        .prepare("SELECT id FROM t WHERE id = $1")
        .await
        .unwrap();
    for i in 1..=3_i64 {
        let result = client.query_prepared(&stmt, &[i.into()]).await.unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0].get_i64(0).unwrap(), Some(i));
    }

    client.close().await.unwrap();
    server.abort();
}

/// Bulk insert via `execute_many`: one prepared statement, many parameter sets, each set's
/// affected-row count returned.
#[tokio::test]
async fn execute_many_bulk_insert() {
    let (host, port, server) = boot_server().await;
    let config = Config::new(host, port, "u", "nusadb");
    let mut client = Client::connect(&config).await.unwrap();

    client
        .simple_query("CREATE TABLE t (id INT NOT NULL, name TEXT)")
        .await
        .unwrap();
    let sets: Vec<Vec<Param>> = vec![
        vec![1_i64.into(), "a".into()],
        vec![2_i64.into(), "b".into()],
        vec![3_i64.into(), "c".into()],
    ];
    let counts = client
        .execute_many("INSERT INTO t VALUES ($1, $2)", &sets)
        .await
        .unwrap();
    assert_eq!(counts, vec![Some(1), Some(1), Some(1)]);

    let total = client.simple_query("SELECT count(*) FROM t").await.unwrap();
    assert_eq!(total.rows[0].get_i64(0).unwrap(), Some(3));

    // An empty batch is a no-op.
    let empty = client
        .execute_many("INSERT INTO t VALUES ($1, $2)", &[])
        .await
        .unwrap();
    assert!(empty.is_empty());

    client.close().await.unwrap();
    server.abort();
}

/// Explicit transactions over the wire: a rolled-back insert leaves no row, a committed one
/// persists, and the `transaction` helper commits on success and rolls back on error.
#[tokio::test]
async fn explicit_transactions_commit_and_rollback() {
    let (host, port, server) = boot_server().await;
    let config = Config::new(host, port, "u", "nusadb");
    let mut client = Client::connect(&config).await.unwrap();

    client
        .simple_query("CREATE TABLE t (id INT NOT NULL)")
        .await
        .unwrap();

    // Rolled-back insert leaves no row.
    client.begin().await.unwrap();
    client
        .execute("INSERT INTO t VALUES (1)", &[])
        .await
        .unwrap();
    client.rollback().await.unwrap();
    assert_eq!(
        client
            .simple_query("SELECT id FROM t")
            .await
            .unwrap()
            .rows
            .len(),
        0
    );

    // Committed insert persists.
    client.begin().await.unwrap();
    client
        .execute("INSERT INTO t VALUES (2)", &[])
        .await
        .unwrap();
    client.commit().await.unwrap();
    let after = client.simple_query("SELECT id FROM t").await.unwrap();
    assert_eq!(after.rows.len(), 1);
    assert_eq!(after.rows[0].get_i64(0).unwrap(), Some(2));

    // The `transaction` helper commits on Ok.
    client
        .transaction(async |c| {
            c.execute("INSERT INTO t VALUES (3)", &[]).await?;
            Ok(())
        })
        .await
        .unwrap();
    assert_eq!(
        client
            .simple_query("SELECT id FROM t")
            .await
            .unwrap()
            .rows
            .len(),
        2
    );

    // The `transaction` helper rolls back on Err.
    let result: nusadb_libnusa::Result<()> = client
        .transaction(async |c| {
            c.execute("INSERT INTO t VALUES (4)", &[]).await?;
            Err(nusadb_libnusa::Error::Protocol("abort".to_owned()))
        })
        .await;
    assert!(result.is_err());
    assert_eq!(
        client
            .simple_query("SELECT id FROM t")
            .await
            .unwrap()
            .rows
            .len(),
        2
    );

    client.close().await.unwrap();
    server.abort();
}

#[tokio::test]
async fn server_error_is_reported_and_connection_survives() {
    let (host, port, server) = boot_server().await;
    let config = Config::new(host, port, "u", "nusadb");
    let mut client = Client::connect(&config).await.unwrap();

    let err = client
        .simple_query("SELECT * FROM ghost")
        .await
        .unwrap_err();
    match err {
        nusadb_libnusa::Error::Server { code, .. } => assert_eq!(code.len(), 5),
        other => panic!("expected a server error, got {other:?}"),
    }

    // The connection is still usable after a server error.
    let ok = client.simple_query("SELECT 1").await.unwrap();
    assert_eq!(ok.rows.len(), 1);

    client.close().await.unwrap();
    server.abort();
}

#[tokio::test]
async fn backend_key_is_captured() {
    let (host, port, server) = boot_server().await;
    let config = Config::new(host, port, "u", "nusadb");
    let client = Client::connect(&config).await.unwrap();
    let key = client.backend_key().expect("server sends BackendKeyData");
    assert!(key.pid > 0);
    // A cancel with the captured key is accepted (a no-op when nothing is running).
    Client::cancel(&config, key).await.unwrap();
    client.close().await.unwrap();
    server.abort();
}

#[tokio::test]
async fn pool_reuses_connections() {
    let (host, port, server) = boot_server().await;
    let config = Config::new(host, port, "u", "nusadb");
    let pool = Pool::new(config, 2);

    let mut c = pool.get().await.unwrap();
    c.simple_query("CREATE TABLE t (id INT NOT NULL)")
        .await
        .unwrap();
    c.execute("INSERT INTO t VALUES (7)", &[]).await.unwrap();
    drop(c); // returned to the pool here
    assert_eq!(pool.idle_count(), 1);

    let mut c = pool.get().await.unwrap();
    let r = c.simple_query("SELECT id FROM t").await.unwrap();
    assert_eq!(r.rows[0].get_i64(0).unwrap(), Some(7));
    drop(c); // same connection reused and returned
    assert_eq!(pool.idle_count(), 1);

    server.abort();
}

/// SCRAM-SHA-256 end to end against an authenticated server.
#[tokio::test]
async fn scram_authentication() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine: Arc<dyn StorageEngine> = Arc::new(BtreeEngine::new());

    let store = AuthStore::from_passwords([("alice", "secret")]).unwrap();
    let server_config = ServerConfig {
        auth: Some(Arc::new(store)),
        ..ServerConfig::default()
    };
    let server = tokio::spawn(serve_with_shutdown(
        listener,
        engine,
        server_config,
        std::future::pending::<()>(),
    ));

    let host = addr.ip().to_string();
    // Correct password authenticates.
    let good = Config::new(host.clone(), addr.port(), "alice", "nusadb").password("secret");
    let mut client = Client::connect(&good).await.unwrap();
    assert_eq!(client.simple_query("SELECT 1").await.unwrap().rows.len(), 1);
    client.close().await.unwrap();

    // Wrong password is rejected.
    let bad = Config::new(host, addr.port(), "alice", "nusadb").password("wrong");
    assert!(Client::connect(&bad).await.is_err());

    server.abort();
}
