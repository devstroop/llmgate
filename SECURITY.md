# Security Policy

## Reporting a vulnerability

Please do **not** open a public issue for security vulnerabilities. Report
privately so the maintainers can fix and release before details are public.

- **Email**: akash@devstroop.com
- **Subject prefix**: `[model-adapter security]`

You can expect an acknowledgement within 48 hours and a status update within
5 business days. Once a fix is available, a public advisory will describe the
issue and the affected versions.

## Scope

The following are in scope:

- Remote code execution, denial of service, or information disclosure through
  the gateway (request parsing, streaming, upstream forwarding).
- Authentication bypass (auth middleware, API-key handling).
- Secret leakage (upstream credentials, client keys) in logs or responses.

The following are out of scope:

- Vulnerabilities in upstream LLM providers themselves.
- Client-side usage of the gateway outside its intended deployment.

## Supported versions

Security fixes are applied to the current `main` and released in the next
version. There are no long-term-support branches at this time.

## Hardening notes

- Inbound request bodies are capped at 16 MiB.
- SSE framing buffers are capped at 64 MiB.
- API-key comparison is constant-time.
- Do not deploy with `[auth] api_keys = []` on an untrusted network — an empty
  key list disables authentication entirely.
