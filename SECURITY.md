# Security Policy

## Supported Versions

This is a demo/portfolio project maintained on a best-effort basis. Only the
latest release on the `main` branch receives security fixes.

| Version | Supported |
| --- | --- |
| latest (main) | ✅ |
| older tags | ❌ |

## Reporting a Vulnerability

If you find a security issue (e.g., a dependency vulnerability, an unsafe
deserialization path, or an input-validation bypass in the Gradio demo),
please **do not** open a public issue. Instead:

1. Use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability) feature on this repository (Security tab → "Report a vulnerability"), or
2. Contact the maintainer directly via the email on their GitHub profile.

Please include:
- A description of the issue and its potential impact.
- Steps to reproduce (a minimal example is ideal).
- Any suggested remediation, if you have one.

You should expect an initial response within 5 business days.

## Scope Notes

- This project loads MNIST from OpenML and serves a local Gradio demo; it
  has no authentication layer, user accounts, or persistent user data store.
  Reports about those areas are out of scope (they don't exist).
- Dependency vulnerabilities are tracked via GitHub Dependabot
  (`.github/dependabot.yml`) and `pip-audit` in CI.
