//! Brave Search API provider for `web_search`.
//!
//! Owns request shaping against Brave's `GET /res/v1/web/search` and the
//! mapping of its JSON envelope onto the provider-neutral
//! [`WebSearchProviderResponse`].
//!
//! Unlike You.com, Brave has no server-side domain filtering: `include_domains`
//! and `exclude_domains` are post-filtered client-side by host suffix on a
//! label boundary. `count` is capped at 20 (clamped, never an error) and the
//! free-tier rate limit is ~1 QPS, so the provider's own concurrency ceiling is
//! 1.
//!
//! Transport rule: the gateway's `reqwest` client is built without gzip
//! support (see `crates/agentic-server-core/Cargo.toml`), so no
//! `Accept-Encoding: gzip` header is sent or expected.

use std::fmt;
use std::future::Future;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;

use serde::Deserialize;

use super::args::{DomainFilter, Freshness, WebSearchArguments, clean_string, clean_vec, validate_count};
use super::{
    WebSearchProvider, WebSearchProviderMetadata, WebSearchProviderResponse, WebSearchResult, null_as_default,
    read_response_limited,
};
use crate::config::WebSearchProviderKind;
use crate::tool::handler::ToolError;
use crate::types::tools::{WebSearchContextSize, WebSearchToolParam};

pub(crate) const BRAVE_API_KEY: &str = WebSearchProviderKind::Brave.default_api_key_env();
pub(crate) const BRAVE_API_BASE_URL: &str = "BRAVE_API_BASE_URL";
/// Default Brave Search API base URL used when no base URL is configured.
pub(crate) const BRAVE_DEFAULT_BASE_URL: &str = "https://api.search.brave.com";
/// Free-tier developer cap on results per section; requested counts above it
/// are clamped rather than rejected, since the model cannot predict provider
/// limits.
const BRAVE_MAX_COUNT: u8 = 20;
/// Free-tier rate limit of ~1 QPS: the provider never runs more than one
/// request in flight.
const BRAVE_MAX_CONCURRENT_REQUESTS: NonZeroUsize = NonZeroUsize::new(1).expect("brave ceiling is nonzero");

/// Provider credential whose `Debug` output never contains the secret.
#[derive(Clone)]
pub(crate) struct ApiKey(pub String);

impl fmt::Debug for ApiKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ApiKey(<redacted>)")
    }
}

#[derive(Debug, Clone)]
pub(crate) struct BraveSearchProvider {
    client: Arc<reqwest::Client>,
    api_key: Option<ApiKey>,
    base_url: Option<String>,
}

impl BraveSearchProvider {
    /// Builds a provider from optional environment-style values: a blank key
    /// counts as unset and fails at execution time. A blank base URL falls
    /// back to [`BRAVE_DEFAULT_BASE_URL`].
    pub(crate) fn from_values(client: Arc<reqwest::Client>, api_key: Option<String>, base_url: Option<String>) -> Self {
        let api_key = api_key
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty())
            .map(ApiKey);
        let base_url = base_url
            .and_then(|value| clean_base_url(&value))
            .or_else(|| Some(BRAVE_DEFAULT_BASE_URL.to_owned()));
        Self {
            client,
            api_key,
            base_url,
        }
    }
}

impl WebSearchProvider for BraveSearchProvider {
    fn search<'a>(
        &'a self,
        query: &'a str,
        args: &'a WebSearchArguments,
        config: &'a WebSearchToolParam,
    ) -> Pin<Box<dyn Future<Output = Result<WebSearchProviderResponse, ToolError>> + Send + 'a>> {
        Box::pin(async move {
            let api_key = self
                .api_key
                .as_ref()
                .ok_or_else(|| ToolError::Config(format!("{BRAVE_API_KEY} must be set to use the web_search tool")))?;
            let base_url = self.base_url.as_deref().ok_or_else(|| {
                ToolError::Config(format!("{BRAVE_API_BASE_URL} must be set to use the web_search tool"))
            })?;
            let request = BraveSearchRequest::from_args_and_config(query, args, config)?;
            let resp = self
                .client
                .get(format!("{base_url}/res/v1/web/search"))
                .query(&request.query_params())
                .header("X-Subscription-Token", &api_key.0)
                .send()
                .await
                .map_err(|e| ToolError::Execution(format!("Brave search request failed: {e}")))?;

            if !resp.status().is_success() {
                let status = resp.status();
                // Surface the upstream rate-limit hint so operators can back off;
                // no automatic retry (Phase 2 scope).
                let retry_after = resp
                    .headers()
                    .get("retry-after")
                    .and_then(|value| value.to_str().ok())
                    .map(str::to_owned);
                let body = read_response_limited(resp, WebSearchProviderKind::Brave)
                    .await
                    .unwrap_or_default();
                let retry_note = retry_after.map_or_else(String::new, |value| format!(" (retry after {value}s)"));
                return Err(ToolError::Execution(format!(
                    "Brave search returned {status}{retry_note}: {body}"
                )));
            }

            let response_text = read_response_limited(resp, WebSearchProviderKind::Brave).await?;
            let response: BraveSearchResponse = serde_json::from_str(&response_text)
                .map_err(|e| ToolError::Execution(format!("Brave search returned invalid JSON: {e}")))?;
            Ok(response.into_provider_response(&request.query, &request.domain_filter))
        })
    }

    fn max_concurrent_requests(&self) -> Option<NonZeroUsize> {
        Some(BRAVE_MAX_CONCURRENT_REQUESTS)
    }
}

/// Query parameters for Brave's `GET /res/v1/web/search`, derived from the
/// model's arguments and the request-level tool configuration.
#[derive(Debug, PartialEq)]
struct BraveSearchRequest {
    query: String,
    count: Option<u8>,
    freshness: Option<Freshness>,
    country: Option<String>,
    language: Option<String>,
    domain_filter: DomainFilter,
}

impl BraveSearchRequest {
    fn query_params(&self) -> Vec<(String, String)> {
        let mut params = vec![
            ("q".to_owned(), self.query.clone()),
            ("result_filter".to_owned(), "web,news".to_owned()),
        ];
        if let Some(count) = self.count {
            params.push(("count".to_owned(), count.to_string()));
        }
        if let Some(freshness) = &self.freshness {
            let rendered = brave_freshness(freshness);
            if !rendered.is_empty() {
                params.push(("freshness".to_owned(), rendered));
            }
        }
        if let Some(country) = &self.country {
            params.push(("country".to_owned(), country.clone()));
        }
        if let Some(language) = &self.language {
            params.push(("search_lang".to_owned(), language.clone()));
        }
        params
    }

    fn from_args_and_config(
        query: &str,
        args: &WebSearchArguments,
        config: &WebSearchToolParam,
    ) -> Result<Self, ToolError> {
        let count = args
            .count
            .or_else(|| {
                config
                    .search_context_size
                    .map(WebSearchContextSize::default_count)
                    .map(u16::from)
            })
            .map(clamp_count)
            .transpose()?;
        let config_domains = config
            .filters
            .as_ref()
            .and_then(|filters| clean_vec(filters.allowed_domains.as_deref()));
        let config_blocked_domains = config
            .filters
            .as_ref()
            .and_then(|filters| clean_vec(filters.blocked_domains.as_deref()));
        let include_domains = config_domains.or_else(|| args.include_domains.clone());
        let exclude_domains = config_blocked_domains.or_else(|| args.exclude_domains.clone());
        // Brave cannot filter domains server-side; remember them for a
        // client-side post-filter instead of dropping or erroring.
        let domain_filter = DomainFilter::new(include_domains.as_deref(), exclude_domains.as_deref());
        if args.boost_domains.is_some() {
            tracing::debug!("web_search boost_domains ignored: Brave does not support boosting");
        }
        if args.livecrawl.is_some() || args.livecrawl_formats.is_some() || args.crawl_timeout.is_some() {
            tracing::debug!("web_search livecrawl arguments ignored: You.com-specific, unsupported by Brave");
        }
        let country = config
            .user_location
            .as_ref()
            .and_then(|location| clean_string(location.country.as_deref()))
            .or_else(|| args.country.clone())
            .map(|value| value.to_ascii_uppercase());

        Ok(Self {
            query: query.trim().to_owned(),
            count,
            freshness: args.freshness,
            country,
            language: args.language.clone(),
            domain_filter,
        })
    }
}

/// Renders a typed freshness filter in Brave's syntax (`pd`/`pw`/`pm`/`py`).
///
/// A custom date range has no Brave equivalent, so it renders to an empty
/// string and is omitted from the query; the operator-visible note is a
/// caller concern.
fn brave_freshness(freshness: &Freshness) -> String {
    match freshness {
        Freshness::Day => "pd".to_owned(),
        Freshness::Week => "pw".to_owned(),
        Freshness::Month => "pm".to_owned(),
        Freshness::Year => "py".to_owned(),
        Freshness::Range { .. } => {
            tracing::debug!("web_search freshness date range ignored: Brave only supports pd/pw/pm/py");
            String::new()
        }
    }
}

/// Clamps a requested count to Brave's per-section cap without failing the
/// call: models cannot be expected to know the provider's limit.
fn clamp_count(count: u16) -> Result<u8, ToolError> {
    let valid = validate_count(count)?;
    Ok(valid.min(BRAVE_MAX_COUNT))
}

fn clean_base_url(value: &str) -> Option<String> {
    let trimmed = value.trim().trim_end_matches('/');
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

/// Brave's `GET /res/v1/web/search` response envelope.
///
/// Forward-compatible: all fields default and unknown keys are tolerated so a
/// Brave API change does not fail the whole search. Web and news result items
/// map straight onto [`WebSearchResult`]; Brave's cosmetic fields
/// (`thumbnail_url`, etc.) and unknown keys are dropped.
#[derive(Debug, Default, Deserialize)]
struct BraveSearchResponse {
    #[serde(default, deserialize_with = "null_as_default")]
    web: BraveResults,
    #[serde(default, deserialize_with = "null_as_default")]
    news: BraveResults,
    #[serde(default, deserialize_with = "null_as_default")]
    query: BraveQuery,
}

#[derive(Debug, Default, Deserialize)]
struct BraveResults {
    #[serde(default, deserialize_with = "null_as_default")]
    results: Vec<BraveResult>,
}

#[derive(Debug, Default, Deserialize)]
struct BraveQuery {
    #[serde(default)]
    search_term: Option<String>,
}

/// One Brave result item. `serde(default)` keeps every field optional so the
/// provider-neutral mapping degrades gracefully.
#[derive(Debug, Default, Deserialize)]
struct BraveResult {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    url: Option<String>,
}

impl BraveSearchResponse {
    fn into_provider_response(self, query: &str, filter: &DomainFilter) -> WebSearchProviderResponse {
        let mut web: Vec<WebSearchResult> = self.web.results.into_iter().map(into_web_search_result).collect();
        let mut news: Vec<WebSearchResult> = self.news.results.into_iter().map(into_web_search_result).collect();
        filter.retain(&mut web);
        filter.retain(&mut news);
        WebSearchProviderResponse {
            web,
            news,
            metadata: WebSearchProviderMetadata {
                provider: WebSearchProviderKind::Brave,
                query: self.query.search_term.unwrap_or_else(|| query.to_owned()),
                search_uuid: None,
                latency: None,
            },
        }
    }
}

fn into_web_search_result(result: BraveResult) -> WebSearchResult {
    WebSearchResult {
        url: result.url.unwrap_or_default(),
        title: result.title,
        description: result.description,
        ..WebSearchResult::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::tools::{WebSearchFilters, WebSearchUserLocation};

    fn args(json: &str) -> WebSearchArguments {
        WebSearchArguments::from_json(json).unwrap()
    }

    #[test]
    fn api_key_debug_is_redacted() {
        let provider = BraveSearchProvider::from_values(
            Arc::new(reqwest::Client::new()),
            Some("super-secret-key".to_owned()),
            Some("https://api.example".to_owned()),
        );
        let rendered = format!("{provider:?}");
        assert!(!rendered.contains("super-secret-key"));
        assert!(rendered.contains("ApiKey(<redacted>)"));
        assert_eq!(format!("{:?}", ApiKey("k".to_owned())), "ApiKey(<redacted>)");
    }

    #[test]
    fn from_values_defaults_base_url_and_treats_blank_credentials_as_unset() {
        let provider = BraveSearchProvider::from_values(Arc::new(reqwest::Client::new()), None, None);
        assert!(provider.api_key.is_none());
        assert_eq!(provider.base_url.as_deref(), Some(BRAVE_DEFAULT_BASE_URL));

        let provider = BraveSearchProvider::from_values(
            Arc::new(reqwest::Client::new()),
            Some("  ".to_owned()),
            Some(" https://api.example/// ".to_owned()),
        );
        assert!(provider.api_key.is_none());
        assert_eq!(provider.base_url.as_deref(), Some("https://api.example"));

        let provider = BraveSearchProvider::from_values(Arc::new(reqwest::Client::new()), Some("k".to_owned()), None);
        assert_eq!(provider.base_url.as_deref(), Some(BRAVE_DEFAULT_BASE_URL));
    }

    #[test]
    fn provider_caps_concurrency_to_one() {
        let provider = BraveSearchProvider::from_values(Arc::new(reqwest::Client::new()), Some("k".to_owned()), None);
        assert_eq!(provider.max_concurrent_requests(), Some(BRAVE_MAX_CONCURRENT_REQUESTS));
    }

    #[test]
    fn request_renders_brave_query_params() {
        let args = args(
            r#"{"query":" rust ","count":5,"freshness":"day","country":"us","language":"en","exclude_domains":["a.example"]}"#,
        );
        let request =
            BraveSearchRequest::from_args_and_config(" rust ", &args, &WebSearchToolParam::default()).unwrap();
        let expected = [
            ("q", "rust"),
            ("result_filter", "web,news"),
            ("count", "5"),
            ("freshness", "pd"),
            ("country", "US"),
            ("search_lang", "en"),
        ]
        .map(|(key, value)| (key.to_owned(), value.to_owned()));
        assert_eq!(request.query_params(), expected);
    }

    #[test]
    fn request_clamps_count_to_brave_cap() {
        let args = args(r#"{"query":"rust","count":50}"#);
        let request = BraveSearchRequest::from_args_and_config("rust", &args, &WebSearchToolParam::default()).unwrap();
        assert_eq!(request.count, Some(BRAVE_MAX_COUNT));
    }

    #[test]
    fn request_drops_unrenderable_date_range_freshness() {
        let args = args(r#"{"query":"rust","freshness":"2024-01-01to2024-02-01"}"#);
        let request = BraveSearchRequest::from_args_and_config("rust", &args, &WebSearchToolParam::default()).unwrap();
        let expected =
            [("q", "rust"), ("result_filter", "web,news")].map(|(key, value)| (key.to_owned(), value.to_owned()));
        assert_eq!(request.query_params(), expected);
    }

    #[test]
    fn request_applies_context_size_default_and_tool_config_overrides() {
        let config = WebSearchToolParam {
            search_context_size: Some(WebSearchContextSize::High),
            filters: Some(WebSearchFilters {
                allowed_domains: Some(vec![" docs.example ".to_owned()]),
                blocked_domains: Some(vec!["bad.example".to_owned()]),
            }),
            user_location: Some(WebSearchUserLocation {
                country: Some(" de ".to_owned()),
                ..WebSearchUserLocation::default()
            }),
        };
        let args = args(r#"{"query":"rust","country":"us","include_domains":["other.example"]}"#);
        let request = BraveSearchRequest::from_args_and_config("rust", &args, &config).unwrap();
        // Config filters win over argument filters; allowlist is hard.
        assert!(request.domain_filter.allows("https://docs.example/page"));
        assert!(!request.domain_filter.allows("https://bad.example/page"));
        assert!(!request.domain_filter.allows("https://other.example/page"));
        assert_eq!(request.country.as_deref(), Some("DE"));
    }

    #[test]
    fn domain_filter_respects_label_boundaries() {
        let filter = DomainFilter::new(Some(&["example.com".to_owned()]), Some(&["a.example.com".to_owned()]));
        assert!(filter.allows("https://example.com/"));
        assert!(filter.allows("https://sub.example.com/x"));
        assert!(!filter.allows("https://notexample.com/"));
        assert!(!filter.allows("https://a.example.com/"));
        // Unparseable URL is rejected whenever any filter is active (fail closed).
        assert!(!filter.allows("not a url"));
    }

    #[test]
    fn response_maps_documented_fields_and_tolerates_nulls() {
        let response: BraveSearchResponse = serde_json::from_str(
            r#"{
                "web": {"results": [
                    {"title": "Rust", "description": "desc", "url": "https://example.com/rust", "unknown_key": 1}
                ]},
                "news": {"results": []},
                "query": {"search_term": "rust"}
            }"#,
        )
        .unwrap();
        let mapped = response.into_provider_response("rust", &DomainFilter::default());
        assert!(mapped.news.is_empty());
        assert_eq!(mapped.metadata.provider, WebSearchProviderKind::Brave);
        assert_eq!(mapped.metadata.query, "rust");
        let result = &mapped.web[0];
        assert_eq!(result.url, "https://example.com/rust");
        assert_eq!(result.title.as_deref(), Some("Rust"));
        assert_eq!(result.description.as_deref(), Some("desc"));
    }

    #[test]
    fn response_applies_domain_post_filter() {
        let response: BraveSearchResponse = serde_json::from_str(
            r#"{
                "web": {"results": [
                    {"title": "A", "url": "https://example.com/a"},
                    {"title": "B", "url": "https://bad.example/b"}
                ]},
                "query": {}
            }"#,
        )
        .unwrap();
        let filter = DomainFilter::new(None, Some(&["bad.example".to_owned()]));
        let mapped = response.into_provider_response("rust", &filter);
        assert_eq!(mapped.web.len(), 1);
        assert_eq!(mapped.web[0].url, "https://example.com/a");
    }

    #[test]
    fn response_tolerates_missing_envelope_sections() {
        let response: BraveSearchResponse = serde_json::from_str("{}").unwrap();
        let mapped = response.into_provider_response("rust", &DomainFilter::default());
        assert!(mapped.web.is_empty());
        assert!(mapped.news.is_empty());
        assert_eq!(mapped.metadata.query, "rust");
    }
}
