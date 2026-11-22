# Changelog

## 0.1.0a1 — preview candidate, unreleased

- Package the existing Qwen3-0.6B LoRA detector as a standalone authenticated
  HTTP service, independent of OpenAIDR and a database.
- Prepare private tenant/key/configuration state using existing hash-verified
  model files, without copying or retraining weights.
- Support loopback startup and require TLS for remote listeners.
- Return model-bound uncalibrated unsafe scores and explicit machine-readable
  unsupported, unavailable, admission and deadline errors.
- Include setup instructions, API examples, provenance/manifests and service
  contract tests. No weights are redistributed.

The original profile and scoring recipe remain unchanged. The optional Q8
ensemble remains experimental and is not selected by standalone preparation.

Verification before standalone extraction included a fresh API dependency
installation, model-extra resolution without installation, controlled error
responses and two functional requests in the existing model environment.
These checks do not establish detector quality, a fully fresh model-runtime
installation or production GPU performance/reliability. No PyPI, container or
public repository release is implied by this candidate.
