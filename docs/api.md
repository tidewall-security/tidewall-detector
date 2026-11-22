# HTTP API reference

The default standalone listener is `http://127.0.0.1:8107`. Remote listeners
require TLS. This document describes schema version 1 and the default
`tidewall.prompt-injection.1` profile; the optional ensemble is experimental.

## Assess a prompt

`POST /v1/classify`

Required headers:

| Header | Value |
| --- | --- |
| `Authorization` | `Bearer <service key>` |
| `X-Tidewall-Tenant-ID` | Canonical UUID from service configuration |
| `Content-Type` | `application/json` |

```json
{
  "schema_version": 1,
  "profile": "tidewall.prompt-injection.1",
  "request_id": "c1654aa5-000c-41d9-871d-ae81464dfd3b",
  "event_kind": "prompt",
  "messages": [{"role": "user", "content": "Explain how a rainbow forms."}],
  "timeout_ms": 1500
}
```

Use a new canonical UUID for every new assessment. Exactly one nonempty user
message is supported. Unknown fields, multi-message conversations, system/tool
roles and other event kinds are rejected rather than silently reduced to text.
The body is bounded at 12,000 bytes before parsing; message content is bounded
at 8,000 UTF-8 bytes and 1,024 tokens after the pinned template. Some reserved
tokenizer control strings are unsupported. Encoded/compressed bodies are not
accepted.

`timeout_ms` accepts integers from 1 to 10,000, but the current service caps the
effective deadline at 1,500 ms. It is not a performance guarantee. Timed-out
physical work may continue until the model returns; admission stays bounded.

## Successful response

HTTP 200 returns these fields:

| Field | Meaning |
| --- | --- |
| `schema_version`, `profile`, `request_id` | Version, selected profile and assessment identity |
| `status` | `classified` |
| `unsafe_score` | Finite score from 0 to 1 |
| `score_kind` | `two_class_softmax_uncalibrated` |
| `model` | Base/adapter/tokenizer revisions, prompt hash, class-token IDs and scoring implementation identity |
| `input_tokens` | Templated input token count |
| `inference_ms` | Time measured around the model scoring call |

The model metadata is documented in [model scope](model-scope.md). The score
is not a calibrated attack probability or severity. A response has no policy
decision, automatic threshold or enforcement action. A caller interpreting a
score must separately choose its policy and validate resulting behavior.

## Errors

Application-generated errors contain `schema_version`, `profile`, `status: error`, `code`,
`request_id` (possibly null) and `retryable`. There is no substitute safe score.

| HTTP | Code | Retryable |
| --- | --- | --- |
| 401 | `unauthenticated` | false |
| 403 | `tenant_mismatch` | false |
| 409 | `request_conflict` | false |
| 413 | `input_too_large` | false |
| 422 | `invalid_request`, `unsupported_input` | false |
| 503 | `model_unavailable`, `capacity_exceeded` | true |
| 504 | `deadline_exceeded` | true |

This envelope does not cover HTTP server or intermediary failures. For example,
Uvicorn can reject requests above its connection/concurrency limit with a
plain-text HTTP 503 before the application runs. Clients must handle transport
errors, non-JSON bodies and invalid/missing response fields as unavailable
inference, never as a safe classification. Do not assume every HTTP error body
can be decoded into the application schema.

There is no durable assessment queue. Successful responses may be replayed
from a bounded in-memory cache when UUID, tenant, profile and content match.
Reusing an admitted UUID with different content is rejected while its record
is retained. Do not rely on replay state surviving cache expiry or restarts.
Use bounded retries and define failure behavior in the caller. A retryable
error does not mean an action was blocked or that retrying will succeed.

## Health

- `GET /health/live`: HTTP 200, `{"status":"alive"}`.
- `GET /health/ready`: HTTP 200 with `status: ready`, or HTTP 503 with
  `status: loading`/`unavailable`; also includes the selected profile.

These endpoints are unauthenticated. Liveness does not imply model readiness.
Connection failure from a stopped service is a transport error, not an HTTP
classification response. Check HTTP status and the response schema rather than
using an empty result or failed request as evidence that a prompt is safe.
