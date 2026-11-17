# Security

## Reporting a vulnerability

Vulnerability-response owner: **Parallel Security**, through the Vulnerability Disclosure Program.

Report security issues in this repository to
[parallel-web-systems-vdp-ess@submit.bugcrowd.com](mailto:parallel-web-systems-vdp-ess@submit.bugcrowd.com), Parallel's Vulnerability
Disclosure Program. Anyone may report through it. Reports are triaged and tracked to
resolution.

Include a description of the issue, the affected version or commit, and steps to
reproduce it. Do not include credentials, restricted benchmark questions, or raw run
traces. Do not post exploit details in public issues.

## Scope

This repository is in scope for the Vulnerability Disclosure Program, which pays no monetary
reward. That covers the repository itself, the code in it, and the published package. It is
**not** in scope for Parallel's private managed bug bounty, which covers a fixed list of
Parallel-operated hosts.

This repository does not grant authorization to test Parallel services. This is a client
application that calls the Parallel API with your own credentials; testing it does not
authorize testing of `api.parallel.ai` or any other Parallel system. Findings in the Parallel
service itself belong to the programs above under their own scope and rules.

Two classes of report are closed without further review: automated scanner output with no
demonstrated impact, and theoretical vectors with no demonstrated exploitability.

## Operating this harness safely

Treat provider credentials and all raw run and preparation artifacts as private. Run
directories can contain benchmark questions, reference answers and raw provider responses.
Review result files before sharing them; a summary does not establish redistribution
rights for benchmark content. Rotate any exposed credentials with the provider that issued
them and preserve sanitized incident evidence.

The [contribution instructions](CONTRIBUTING.md) install a pinned pre-commit secret
scanner. CI scans both the checkout and its commit history, so skipping local hooks
or removing a credential in a later commit does not skip that check. Scanning is
heuristic: it does not establish that a file or Git history
contains no credentials. Keep real credentials out of the checkout and use the
existing public-content and distribution checks before publishing artifacts.
