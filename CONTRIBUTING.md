# Contributing

This repository contains the standalone Tidewall detector service. Agent
instrumentation, guard policy enforcement and dashboards belong to their
respective projects. The service's public contract is the versioned HTTP API
and documented command-line setup/startup paths; bundled research helpers are
not a promise of a supported evaluation workflow.

## Develop and check

Use Python 3.11 or newer. From this repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q tests
```

The tests require local loopback sockets for listener validation. They use
tiny artifacts and controllable scorers, so neither model weights, GPU packages
nor a cloud account is required. The original class-token recipe hashes are
checked against packaged provenance. No adjacent Tidewall/OpenAIDR checkout
should be required.

Build distributions with the pinned build backend:

```sh
python -m pip install 'build==1.3.0'
python -m build
```

Keep fixes focused and include a regression test for changed behavior. Preserve
schema validation, authentication, tenant binding, explicit unavailable/unsupported
responses, pinned model identity and the distinction between classification and
policy enforcement. Do not label raw scores calibrated probabilities.

Model/profile changes require separate evidence and review. Do not silently
update artifacts, tokenizer, prompt, scoring recipe or thresholds as part of
a service fix. Never check in credentials, model weights, private transcripts
or evaluation datasets without explicit rights/privacy review. Remove private
content and credentials from bug reports. Coordinate sensitive security reports
privately with the maintainers before sharing reproduction details publicly.
Use the reporting route in [SECURITY.md](SECURITY.md) for vulnerabilities.

Keep discussion respectful and focused on reproducible technical evidence.
Critique code and claims rather than people. Maintainers may moderate abuse,
harassment, unsolicited private information and unsafe public disclosures.

Contributions use Apache-2.0, the repository's existing licence. No copyright
assignment or separate contributor agreement is introduced by these instructions.

## Release review

Before publishing, verify source/wheel contents and notices, build from a clean
commit, run the service suite, install the built artifact outside the checkout,
and record the supported runtime/platform and known limitations. Runtime tests,
model-quality evaluations and hardware qualification are separate evidence.
Release notes must identify any unverified installation or deployment claims.
