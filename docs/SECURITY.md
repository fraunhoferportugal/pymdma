# Security Policy

## Guidelines

- Never store credentials as code/config. See [Loading environment variables](LOADING_ENV_VARIABLES.md)
  - Passwords in your publicly available code can easily get into the wrong hands, which is why it's best to
    avoid putting credentials into your repository in the first place
  - [Keycloak](https://www.keycloak.org/) is used for identity and access management in modern applications and services.
  - [Git-secrets](https://github.com/awslabs/git-secrets) statically analyzes your commits via a pre-commit git kook to
    ensure you're not pushing any passwords or sensitive information into your Bitbucket repository.
    Commits are rejected if the tool matches any of the configured regular expression patterns that indicate that sensitive
    information has been stored improperly.
- Remove sensitive data from your files and Bitbucket history.
  - If you commit sensitive data, such as a password or SSH key into a git repository, you can remove it from the history.
    To entirely remove unwanted files from a repository's history you can use either the `git filter-repo` tool or the
    [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/) open source tool.
- Access Control
  - Never let Bitbucket users share accounts/passwords
  - Make sure you diligently revoke access from Bitbucket users who are no longer working with you
- Report vulnerabilities
  - Run [Safety](https://github.com/pyupio/safety) and [Bandit](https://bandit.readthedocs.io/en/latest/) to find new
    vulnerabilities. [Trivy](https://github.com/aquasecurity/trivy) scans vulnerabilities in container images.

<!-- ## Supported Versions

This project is under active development, and we do our best to support the latest versions.

| Version | Supported          |
| ------- | ------------------ |
| latest  |                    | -->

## Reporting a Vulnerability or Security Issues

> Do not open issues that might have security implications!
> It is critical that security related issues are reported privately so we have time to address them before they become
> public knowledge.

Vulnerabilities can be reported by emailing core members:

- Fraunhofer AICOS \[no-reply@fraunhofer.pt\](mailto:no-reply@fraunhofer.pt)

Please include the requested information listed below (as much as you can provide) to help us better understand the
nature and scope of the possible issue:

- Type of issue (e.g. buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Environment (e.g. Linux / Windows / macOS)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

This information will help us triage your report more quickly.

## Preferred Languages

We prefer all communications to be in English.

## References

- [5 tips to keep your code secure](https://bitbucket.org/blog/keep-your-code-secure)
- [Removing sensitive data from a repository](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)
