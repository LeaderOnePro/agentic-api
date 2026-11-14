//! Replays the recorded Files API and Responses `file_id` lifecycles through the HTTP gateway.
//!
//! The recordings live in `agentic-server-core/tests/cassettes/text_files/responses/files-api-*.yaml`.
//! Each replay uploads the recorded multipart bytes, follows the recorded metadata, content, and delete
//! turns, and sends the recorded Responses turns with recorded file and response IDs remapped to the IDs
//! this gateway generated. A mock upstream returns the recorded answer, so these tests check the
//! gateway's file resolution, error mapping, and client-facing persistence rather than model output.

#[allow(dead_code)]
mod common;

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use agentic_core::executor::ExecutionContext;
use axum::Router;
use axum::body::Bytes;
use axum::response::IntoResponse;
use axum::routing::post;
use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use http::StatusCode;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;

const DIR: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../agentic-server-core/tests/cassettes/text_files/responses"
);
const TEXT: &str = "files-api-text-gateway-Qwen-Qwen2.5-3B-Instruct.yaml";
const IMAGE: &str = "files-api-image-gateway-Qwen-Qwen2.5-VL-3B-Instruct.yaml";
const ERRORS: &str = "files-api-errors-gateway-Qwen-Qwen2.5-3B-Instruct.yaml";

fn load(name: &str) -> Value {
    let path = format!("{DIR}/{name}");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|error| panic!("read {path}: {error}"));
    serde_yml::from_str(&text).unwrap_or_else(|error| panic!("parse {path}: {error}"))
}

fn turns(cassette: &Value) -> &[Value] {
    cassette["turns"].as_array().expect("cassette turns")
}

fn turn<'a>(cassette: &'a Value, filename: &str) -> &'a Value {
    turns(cassette)
        .iter()
        .find(|turn| turn["filename"] == filename)
        .unwrap_or_else(|| panic!("cassette turn {filename}"))
}

fn method(turn: &Value) -> &str {
    turn["request"]["method"].as_str().expect("request method")
}

fn path(turn: &Value) -> &str {
    turn["request"]["path"].as_str().expect("request path")
}

fn expected_status(turn: &Value) -> StatusCode {
    let status = u16::try_from(turn["response"]["status_code"].as_u64().expect("status code")).expect("u16 status");
    StatusCode::from_u16(status).expect("valid status code")
}

fn request_bytes(turn: &Value) -> Option<Vec<u8>> {
    turn["request"]["body_base64"]
        .as_str()
        .map(|encoded| STANDARD.decode(encoded).expect("base64 request body"))
}

fn response_bytes(turn: &Value) -> Vec<u8> {
    STANDARD
        .decode(turn["response"]["body_base64"].as_str().expect("base64 response body"))
        .expect("base64 response body")
}

fn recorded_content(cassette: &Value) -> Vec<u8> {
    response_bytes(turn(cassette, "t3"))
}

fn recorded_id(turn: &Value) -> &str {
    turn["response"]["body"]["id"].as_str().expect("recorded ID")
}

fn output_text(body: &Value) -> String {
    body["output"]
        .as_array()
        .into_iter()
        .flatten()
        .flat_map(|item| item["content"].as_array().into_iter().flatten())
        .filter(|part| part["type"] == "output_text")
        .filter_map(|part| part["text"].as_str())
        .collect()
}

fn remap_ids(value: &mut Value, ids: &HashMap<String, String>) {
    match value {
        Value::String(text) => {
            if let Some(replacement) = ids.get(text.as_str()) {
                *text = replacement.clone();
            }
        }
        Value::Array(items) => items.iter_mut().for_each(|item| remap_ids(item, ids)),
        Value::Object(fields) => fields.values_mut().for_each(|field| remap_ids(field, ids)),
        Value::Null | Value::Bool(_) | Value::Number(_) => {}
    }
}

fn remap_path(path: &str, ids: &HashMap<String, String>) -> String {
    path.split('/')
        .map(|segment| ids.get(segment).cloned().unwrap_or_else(|| segment.to_owned()))
        .collect::<Vec<_>>()
        .join("/")
}

fn assert_no_file_reference(request: &Value) {
    assert!(
        request.get("previous_response_id").is_none(),
        "the gateway rehydrates history itself: {request}"
    );
    let encoded = request.to_string();
    assert!(
        !encoded.contains("\"file_id\""),
        "the model must receive the resolved file: {encoded}"
    );
    assert!(
        !encoded.contains("\"file_data\""),
        "the model must receive the resolved file: {encoded}"
    );
}

async fn stored_parts(exec_ctx: &ExecutionContext) -> Vec<Value> {
    let pool = exec_ctx.storage_pool().expect("storage pool");
    let rows: Vec<String> = sqlx::query_scalar("SELECT data FROM items")
        .fetch_all(pool)
        .await
        .expect("stored items");
    rows.iter()
        .flat_map(|row| {
            let item: Value = serde_json::from_str(row).expect("stored item JSON");
            item["content"].as_array().cloned().unwrap_or_default()
        })
        .collect()
}

struct MockUpstream {
    url: String,
    requests: Arc<Mutex<Vec<Value>>>,
    task: JoinHandle<()>,
}

impl Drop for MockUpstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn mock_upstream(responses: Vec<Value>) -> MockUpstream {
    let responses = Arc::new(Mutex::new(VecDeque::from(responses)));
    let requests = Arc::new(Mutex::new(Vec::new()));
    let route_responses = Arc::clone(&responses);
    let route_requests = Arc::clone(&requests);
    let app = Router::new().route(
        "/v1/responses",
        post(move |body: Bytes| {
            let responses = Arc::clone(&route_responses);
            let requests = Arc::clone(&route_requests);
            async move {
                let request: Value = serde_json::from_slice(&body).expect("upstream request JSON");
                requests.lock().await.push(request);
                let response = responses.lock().await.pop_front();
                match response {
                    Some(response) => axum::Json(response).into_response(),
                    None => (
                        StatusCode::INTERNAL_SERVER_ERROR,
                        axum::Json(json!({"error": {"message": "mock upstream response queue exhausted"}})),
                    )
                        .into_response(),
                }
            }
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind mock upstream");
    let address = listener.local_addr().expect("mock upstream address");
    let task = tokio::spawn(async move { axum::serve(listener, app).await.expect("serve mock upstream") });
    MockUpstream {
        url: format!("http://{address}"),
        requests,
        task,
    }
}

struct Gateway {
    url: String,
    exec_ctx: Arc<ExecutionContext>,
    task: JoinHandle<()>,
    _db: tempfile::TempDir,
    _files: tempfile::TempDir,
}

impl Drop for Gateway {
    fn drop(&mut self) {
        self.task.abort();
    }
}

async fn start_gateway(upstream_url: &str) -> Gateway {
    let db = tempfile::tempdir().expect("temp database directory");
    let files = tempfile::tempdir().expect("temp files directory");
    let mut config = common::test_config(upstream_url);
    config.openai_api_key = None;
    config.db_url = Some(format!("sqlite://{}", db.path().join("gateway.db").display()));
    config.tools.file_search.files_storage_dir = Some(files.path().to_owned());
    let mut state = common::test_state(&config);
    state.exec_ctx = Arc::new(ExecutionContext::from_config(&config).await.expect("execution context"));
    let exec_ctx = Arc::clone(&state.exec_ctx);
    let (url, task) = common::spawn_gateway(state).await;
    Gateway {
        url,
        exec_ctx,
        task,
        _db: db,
        _files: files,
    }
}

async fn replay(cassette: &Value, expected_upstream_calls: usize) -> (Gateway, MockUpstream, HashMap<String, String>) {
    let responses = turns(cassette)
        .iter()
        .filter(|turn| path(turn) == "/v1/responses" && expected_status(turn) == StatusCode::OK)
        .map(|turn| turn["response"]["body"].clone())
        .collect::<Vec<_>>();
    assert_eq!(responses.len(), expected_upstream_calls);
    let upstream = mock_upstream(responses).await;
    let gateway = start_gateway(&upstream.url).await;
    let client = reqwest::Client::new();
    let mut file_ids = HashMap::new();
    let mut response_ids = HashMap::new();
    for turn in turns(cassette) {
        replay_turn(&client, &gateway.url, turn, &mut file_ids, &mut response_ids).await;
    }
    assert_eq!(
        upstream.requests.lock().await.len(),
        expected_upstream_calls,
        "rejected files must not reach inference"
    );
    (gateway, upstream, file_ids)
}

async fn replay_turn(
    client: &reqwest::Client,
    gateway_url: &str,
    turn: &Value,
    file_ids: &mut HashMap<String, String>,
    response_ids: &mut HashMap<String, String>,
) {
    let method = method(turn);
    let raw_path = path(turn);
    let path = remap_path(raw_path, file_ids);
    let url = format!("{gateway_url}{path}");
    let mut request = match method {
        "GET" => client.get(url),
        "POST" => client.post(url),
        "DELETE" => client.delete(url),
        other => panic!("unsupported method {other}"),
    };
    if let Some(body) = request_bytes(turn) {
        let content_type = turn["request"]["headers"]["content-type"]
            .as_str()
            .expect("request content type");
        request = request.header("content-type", content_type).body(body);
    } else if raw_path == "/v1/responses" {
        let mut body = turn["request"]["body"].clone();
        remap_ids(&mut body, response_ids);
        remap_ids(&mut body, file_ids);
        request = request.json(&body);
    }
    let response = request.send().await.expect("gateway request");
    let status = response.status();
    assert_eq!(
        status,
        expected_status(turn),
        "{} {raw_path} ({})",
        method,
        turn["filename"]
    );
    let expected = &turn["response"];
    if !status.is_success() {
        let actual: Value = response.json().await.expect("error JSON");
        assert_eq!(actual, expected["body"], "{}", turn["filename"]);
        return;
    }
    if let Some(encoded) = expected["body_base64"].as_str() {
        let actual = response.bytes().await.expect("response bytes");
        assert_eq!(
            actual.as_ref(),
            STANDARD.decode(encoded).expect("expected bytes").as_slice(),
            "{}",
            turn["filename"]
        );
        return;
    }
    let actual: Value = response.json().await.expect("JSON response");
    match (method, raw_path) {
        ("POST", "/v1/files") => assert_upload(&actual, expected, file_ids),
        ("GET", path) if path.starts_with("/v1/files/") => assert_file_metadata(&actual, expected, file_ids),
        ("DELETE", path) if path.starts_with("/v1/files/") => assert_delete(&actual, expected, file_ids),
        ("POST", "/v1/responses") => assert_response(&actual, expected, response_ids),
        (method, path) => panic!("unhandled {method} {path}"),
    }
}

fn assert_upload(actual: &Value, expected: &Value, file_ids: &mut HashMap<String, String>) {
    let recorded_id = expected["body"]["id"].as_str().expect("recorded file ID");
    let actual_id = actual["id"].as_str().expect("uploaded file ID");
    assert!(actual_id.starts_with("file-"), "{actual}");
    assert_ne!(actual_id, recorded_id, "the replay must exercise file ID remapping");
    for field in ["bytes", "filename", "object", "purpose", "status"] {
        assert_eq!(actual[field], expected["body"][field], "upload {field}");
    }
    file_ids.insert(recorded_id.to_owned(), actual_id.to_owned());
}

fn assert_file_metadata(actual: &Value, expected: &Value, file_ids: &HashMap<String, String>) {
    let recorded_id = expected["body"]["id"].as_str().expect("recorded file ID");
    let actual_id = file_ids.get(recorded_id).expect("uploaded file ID");
    assert_eq!(actual["id"].as_str(), Some(actual_id.as_str()));
    for field in ["bytes", "filename", "object", "purpose", "status"] {
        assert_eq!(actual[field], expected["body"][field], "metadata {field}");
    }
}

fn assert_delete(actual: &Value, expected: &Value, file_ids: &HashMap<String, String>) {
    let recorded_id = expected["body"]["id"].as_str().expect("recorded file ID");
    let actual_id = file_ids.get(recorded_id).expect("uploaded file ID");
    assert_eq!(actual["deleted"], true);
    assert_eq!(actual["object"], "file.deleted");
    assert_eq!(actual["id"].as_str(), Some(actual_id.as_str()));
}

fn assert_response(actual: &Value, expected: &Value, response_ids: &mut HashMap<String, String>) {
    let recorded_id = expected["body"]["id"].as_str().expect("recorded response ID");
    let actual_id = actual["id"].as_str().expect("response ID");
    assert!(actual_id.starts_with("resp_"), "{actual}");
    assert_eq!(actual["status"], expected["body"]["status"]);
    assert_eq!(actual["model"], expected["body"]["model"]);
    assert_eq!(actual["store"], expected["body"]["store"]);
    assert_eq!(output_text(actual), output_text(&expected["body"]));
    match expected["body"]["previous_response_id"].as_str() {
        Some(recorded_parent) => {
            let parent = response_ids.get(recorded_parent).expect("parent response ID");
            assert_eq!(actual["previous_response_id"].as_str(), Some(parent.as_str()));
        }
        None => assert!(actual["previous_response_id"].is_null(), "{actual}"),
    }
    response_ids.insert(recorded_id.to_owned(), actual_id.to_owned());
}

#[tokio::test]
async fn files_api_text_reference_replays_through_the_gateway() {
    let cassette = load(TEXT);
    let (gateway, upstream, file_ids) = replay(&cassette, 2).await;
    let notes = recorded_content(&cassette);
    let expected = format!(
        "File: notes.txt\n{}",
        std::str::from_utf8(&notes).expect("notes are UTF-8")
    );
    let requests = upstream.requests.lock().await;
    assert_eq!(requests.len(), 2);
    for request in requests.iter() {
        assert_eq!(
            request["input"][0]["content"][1],
            json!({"type": "input_text", "text": expected}),
            "the model receives the stored file as text"
        );
        assert_no_file_reference(request);
    }
    assert_eq!(
        requests[1]["input"][0]["content"][1],
        json!({"type": "input_text", "text": expected}),
        "a continuation rehydrates the stored file reference"
    );

    let first_file = file_ids.get(recorded_id(turn(&cassette, "t1"))).expect("first upload");
    let stored = stored_parts(&gateway.exec_ctx).await;
    let reference = stored
        .iter()
        .find(|part| part["type"] == "input_file" && part["file_id"].as_str() == Some(first_file.as_str()))
        .expect("stored history keeps the client's file_id");
    assert!(
        reference.get("file_data").is_none(),
        "stored history must not inline the file: {reference}"
    );
    assert!(
        !stored.iter().any(|part| part["text"]
            .as_str()
            .is_some_and(|text| text.contains("Deployment checklist"))),
        "model-facing file text must not be persisted"
    );
}

#[tokio::test]
async fn files_api_image_reference_replays_through_the_gateway() {
    let cassette = load(IMAGE);
    let (gateway, upstream, file_ids) = replay(&cassette, 2).await;
    let image = recorded_content(&cassette);
    let expected_url = format!("data:image/png;base64,{}", STANDARD.encode(&image));
    let requests = upstream.requests.lock().await;
    assert_eq!(requests.len(), 2);
    for request in requests.iter() {
        assert_eq!(
            request["input"][0]["content"][1],
            json!({"type": "input_image", "image_url": expected_url, "detail": "auto"}),
            "the model receives the stored image as a data URL"
        );
        assert_no_file_reference(request);
    }

    let first_file = file_ids.get(recorded_id(turn(&cassette, "t1"))).expect("first upload");
    let stored = stored_parts(&gateway.exec_ctx).await;
    let reference = stored
        .iter()
        .find(|part| part["type"] == "input_image" && part["file_id"].as_str() == Some(first_file.as_str()))
        .expect("stored history keeps the client's file_id");
    assert!(
        reference.get("image_url").is_none(),
        "stored history must not inline the image: {reference}"
    );
}

#[tokio::test]
async fn files_api_upload_and_lookup_errors_replay_without_inference() {
    let cassette = load(ERRORS);
    let (gateway, upstream, _file_ids) = replay(&cassette, 0).await;
    assert!(upstream.requests.lock().await.is_empty());
    let pool = gateway.exec_ctx.storage_pool().expect("storage pool");
    let stored: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM items")
        .fetch_one(pool)
        .await
        .expect("stored item count");
    assert_eq!(stored, 0, "rejected files must not create stored history");
}
