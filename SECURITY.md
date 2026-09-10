# Security Policy

Thank you for helping keep Imparo and its users safe. A report does not need a
complete exploit or a finished fix to be useful.

## Report a vulnerability privately

Email **[fergus@zeraix.com](mailto:fergus@zeraix.com)** with the subject
`Imparo security report`.

Please do not put vulnerability details, exploit code, or sensitive data in a
public issue, discussion, or pull request before we have coordinated disclosure.
Ordinary build failures, feature requests, and performance regressions can use
[public issues](https://github.com/zeraix/imparo/issues/new/choose).

Share what you can:

- Affected commit or release, component, operating system, and backend.
- Expected behavior, observed behavior, and the potential security impact.
- Minimal reproduction steps or a small proof of concept, preferably with
  synthetic inputs rather than private prompts, weights, or cache contents.
- Any relevant configuration or logs with credentials and personal data removed.

We will assess the report, request more information if needed, and coordinate
remediation and disclosure with you. We do not promise a fixed response or patch
deadline. With your permission, we will credit you when publishing a fix.

## Maintained versions

Imparo is under active development. Security fixes target the current public
`main` branch. Reports about older commits are welcome, but we do not maintain a
separate security-backport commitment for each historical snapshot. Include the
exact revision so we can identify the affected code.

## Relevant security boundaries

Reports are welcome for issues in the code in this repository, including:

- Parsing GGUF metadata, tensor sizes, tokenization inputs, and chat templates.
- Native kernel/FFI memory safety and unsafe handling of malformed inputs.
- HTTP request handling and unintended exposure of inference or state operations.
- KV and recurrent-state persistence, file paths, cache identity, and unintended
  disclosure or modification of another conversation's data.
- Program-pack verification, trust policy, installation, and native backend loading.

If a dependency appears responsible, you can still contact us about the impact
on Imparo. We can coordinate with upstream; you do not need to diagnose ownership
before reporting.

## Deployment precautions

- The current server binds to loopback (`127.0.0.1`) by default. Do not expose it
  directly to untrusted networks; add appropriate authentication, access controls,
  transport security, and resource limits for any remote deployment.
- Treat model files, native backend libraries, and kernel program packs as
  security-sensitive inputs. Use trusted sources; a signature does not by itself
  prove a kernel's correctness or make it safe to execute untrusted code.
- Treat persisted inference state and logs as sensitive conversation data. Protect
  their directories and backups with appropriate operating-system permissions.
  Content-addressed reuse is not a substitute for user authentication or tenant
  access control.
- Use only systems and data you own or have permission to test, and avoid testing
  against other people's services or publishing their data.

This policy describes reporting and safe handling, not a security certification
or a guarantee that every deployment configuration is hardened.
