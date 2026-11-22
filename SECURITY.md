# Security reporting

Please report vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/tidewall-security/tidewall-detector/security/advisories/new).
Do not put credentials, private prompts, customer data or exploitable details
in public issues. Use synthetic or redacted examples wherever possible.

Include the affected version/commit, operating system, configuration relevant
to the issue, reproduction steps and expected/actual behavior. Explain the
security boundary involved, such as authentication, tenant binding, model
artifact verification or input admission. Do not include a real bearer key.

This project is an alpha. The latest alpha is the maintenance target; no
long-term support or guaranteed response timeline is offered. A detection
miss or false positive may be useful model feedback, but is not by itself
proof of a service security vulnerability. Report service vulnerabilities
through the private route above and non-sensitive model feedback through a
normal issue.

Serving defaults to loopback. Non-loopback listeners require TLS, and
classification requires a bearer key and matching tenant identifier. Keep
model assets and service state under operator control, and keep credentials
out of source control. Health endpoints expose readiness/liveness without
authentication; restrict network access according to your deployment needs.

The detector returns assessments; enforcement and fail-open/fail-closed
decisions belong to the calling application. See the API reference and model
scope document for input coverage and score limitations.
