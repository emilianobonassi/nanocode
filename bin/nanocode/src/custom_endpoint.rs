//! Point Codex at any OpenAI-compatible Responses endpoint and offer the
//! models it lists.
//!
//! * `NANOCODE_OPENAI_BASE_URL`: base URL of the API, e.g.
//!   `https://api.example.com/v1`. When set and the host did not install its
//!   own OpenAI client, Codex requests go to `<base>/responses` with
//!   `NANOCODE_OPENAI_API_KEY` (or `OPENAI_API_KEY`) as the bearer token.
//! * `NANOCODE_MODELS_URL`: where to read the model list. Defaults to
//!   `<NANOCODE_OPENAI_BASE_URL>/models`. Hosts with their own client (for
//!   example a paying HTTP layer) set only this.
//!
//! The listed IDs replace the built-in Codex models in the picker, `-m` and
//! `nanocode models`, and are sent to the endpoint unchanged.

use crate::host::ModelClients;
use std::sync::atomic::{AtomicBool, Ordering};

const BASE_URL: &str = "NANOCODE_OPENAI_BASE_URL";
const MODELS_URL: &str = "NANOCODE_MODELS_URL";

static SERVED: AtomicBool = AtomicBool::new(false);

/// Whether an endpoint model list was loaded.
pub fn served() -> bool {
    SERVED.load(Ordering::Relaxed)
}

fn env(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty())
}

fn api_key() -> Option<String> {
    env("NANOCODE_OPENAI_API_KEY").or_else(|| env("OPENAI_API_KEY"))
}

pub fn apply(mut clients: ModelClients) -> Result<ModelClients, String> {
    let base = env(BASE_URL).map(|url| url.trim_end_matches('/').to_owned());
    let models_url = env(MODELS_URL).or_else(|| base.as_ref().map(|base| format!("{base}/models")));
    if base.is_none() && models_url.is_none() {
        return Ok(clients);
    }
    nanocode::codex::transport::install_default_rustls_crypto_provider();
    let key = api_key();
    if let Some(url) = models_url {
        let ids = fetch_models(&url, key.as_deref())?;
        if ids.is_empty() {
            return Err(format!("{url} listed no models"));
        }
        nanocode::Model::register_served(&ids);
        SERVED.store(true, Ordering::Relaxed);
    }
    if let Some(base) = base
        && !clients.has_openai()
    {
        let auth = match key {
            Some(key) => nanocode::codex::OpenAiAuth::api_key(key),
            None => nanocode::codex::OpenAiAuth::gateway_managed(),
        };
        clients = clients.openai(move || {
            nanocode::OpenAi::builder(auth.clone())
                .api_base_url(base.clone())
                .transport(nanocode::codex::transport::ResponsesTransport::Https)
        });
    }
    Ok(clients)
}

fn fetch_models(url: &str, key: Option<&str>) -> Result<Vec<String>, String> {
    #[derive(serde::Deserialize)]
    struct List {
        data: Vec<Entry>,
    }
    #[derive(serde::Deserialize)]
    struct Entry {
        id: String,
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    runtime.block_on(async {
        let mut request = reqwest::Client::new()
            .get(url)
            .timeout(std::time::Duration::from_secs(15));
        if let Some(key) = key {
            request = request.bearer_auth(key);
        }
        let response = request
            .send()
            .await
            .map_err(|error| format!("failed to list models from {url}: {error}"))?;
        let status = response.status();
        if !status.is_success() {
            return Err(format!("failed to list models from {url}: HTTP {status}"));
        }
        let list: List = response
            .json()
            .await
            .map_err(|error| format!("{url} did not return an OpenAI model list: {error}"))?;
        let mut ids: Vec<String> = Vec::new();
        for entry in list.data {
            if !ids.contains(&entry.id) {
                ids.push(entry.id);
            }
        }
        Ok(ids)
    })
}
