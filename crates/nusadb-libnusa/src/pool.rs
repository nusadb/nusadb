//! A bounded connection pool.
//!
//! [`Pool::get`] hands out a [`PooledClient`] that reuses an idle connection when one is available
//! and otherwise opens a new one, up to `max_size` live connections. The semaphore bounds the
//! total; surplus `get` calls wait for a connection to be returned. A client is returned to the
//! idle set when its [`PooledClient`] guard drops.
//!
//! Reuse is safe because every query drains the stream to `ReadyForQuery` before returning, so a
//! returned connection is in a clean state. A connection left mid-transaction by the caller is the
//! caller's responsibility (the pool does not reset session state).

use std::ops::{Deref, DerefMut};
use std::sync::{Arc, Mutex, PoisonError};

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::client::{Client, Config};
use crate::error::{Error, Result};

#[derive(Debug)]
struct Inner {
    config: Config,
    idle: Mutex<Vec<Client>>,
    sem: Arc<Semaphore>,
}

/// A bounded pool of [`Client`] connections to one server.
#[derive(Debug, Clone)]
pub struct Pool {
    inner: Arc<Inner>,
}

impl Pool {
    /// Create a pool that opens at most `max_size` connections (clamped to at least 1) using
    /// `config`. Connections are created lazily on demand.
    #[must_use]
    pub fn new(config: Config, max_size: usize) -> Self {
        let permits = max_size.max(1);
        Self {
            inner: Arc::new(Inner {
                config,
                idle: Mutex::new(Vec::new()),
                sem: Arc::new(Semaphore::new(permits)),
            }),
        }
    }

    /// Check out a connection, reusing an idle one or opening a new one. Waits if every connection
    /// is in use.
    ///
    /// # Errors
    /// [`Error::Io`]/[`Error::Tls`]/[`Error::Auth`]/[`Error::Server`] if a new connection must be
    /// opened and the connect/handshake fails.
    pub async fn get(&self) -> Result<PooledClient> {
        let permit = Arc::clone(&self.inner.sem)
            .acquire_owned()
            .await
            .map_err(|_| Error::Protocol("connection pool is closed".to_owned()))?;
        let reused = self.lock_idle().pop();
        let client = match reused {
            Some(client) => client,
            None => Client::connect(&self.inner.config).await?,
        };
        Ok(PooledClient {
            client: Some(client),
            inner: Arc::clone(&self.inner),
            _permit: permit,
        })
    }

    /// The number of idle (checked-in) connections currently held.
    #[must_use]
    pub fn idle_count(&self) -> usize {
        self.lock_idle().len()
    }

    fn lock_idle(&self) -> std::sync::MutexGuard<'_, Vec<Client>> {
        self.inner
            .idle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
    }
}

/// A connection borrowed from a [`Pool`]. Derefs to [`Client`]; returned to the pool on drop.
#[derive(Debug)]
pub struct PooledClient {
    client: Option<Client>,
    inner: Arc<Inner>,
    _permit: OwnedSemaphorePermit,
}

impl PooledClient {
    /// Discard this connection instead of returning it to the pool (e.g. after a protocol error
    /// left it in an unknown state). The pool will open a fresh one next time.
    pub fn discard(mut self) {
        self.client = None;
    }
}

impl Deref for PooledClient {
    type Target = Client;

    fn deref(&self) -> &Client {
        // `client` is `Some` for the whole guard lifetime; it is only taken in `discard`/`drop`,
        // both of which consume `self`.
        self.client
            .as_ref()
            .unwrap_or_else(|| unreachable!("pooled client already taken"))
    }
}

impl DerefMut for PooledClient {
    fn deref_mut(&mut self) -> &mut Client {
        self.client
            .as_mut()
            .unwrap_or_else(|| unreachable!("pooled client already taken"))
    }
}

impl Drop for PooledClient {
    fn drop(&mut self) {
        if let Some(client) = self.client.take() {
            self.inner
                .idle
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(client);
        }
    }
}
