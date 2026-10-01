# Source export

The local tools export the approved committed source tree. They do not push, upload, choose a license or publish a repository. Source export excludes Git history, local migration evidence, data, model weights, checkpoints, research plots and statistical analysis products.

Commit intended changes first, then choose a new destination outside the checkout:

```sh
python scripts/export_release.py --destination /tmp/orbit-core-release
```

The report identifies the source commit and file count. The exporter checks approved paths and file modes, scans committed text, and refuses to replace an existing destination. Scanner findings include categories and paths, not matched credential contents. Files are written into an owned temporary tree before an atomic no-replace directory rename. macOS and Linux have supported rename paths; unsupported platforms fail explicitly.

The scanner no longer has image approvals or figure-manifest handling. The release contains core source, configurations, synthetic examples, usage documentation and contributor infrastructure. Its finite patterns do not establish comprehensive privacy, data rights or legal clearance. Review source rights and history independently before any public release.

The existing public project site remains in the repository. Its nine files are checked against their established SHA-256 payloads and excluded from the core source export. A changed site payload is rejected; new research images receive no exemption. The site is not copied from local research artifacts.

Contributor checks remain in the existing CI and development scripts. Their presence does not establish that a changed tree has passed validation. Run applicable checks after changing layout, import paths, configuration contracts or release scope, and verify GPU/service behavior separately in the intended environment.
