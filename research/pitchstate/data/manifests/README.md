# Dataset manifests

Manifests are the source of truth for dataset identity and split/version
boundaries. They contain metadata only; raw videos, annotations, and model
outputs stay outside version control.

Required principles:

1. Pin a dataset version or download date.
2. Record the source URL and license/access status.
3. Split by match/game, never by neighboring frames.
4. Record any local preprocessing as a new derived manifest version.
5. Do not place credentials, signed URLs, or private download links in a manifest.
6. List match/game IDs and clip IDs in exactly one split; the loader rejects
   duplicates and cross-split leakage.
7. Record annotation families as `source_verified_access_pending` until the
   actual local release has been inspected.

The example manifest is a schema fixture, not a claim that the data has been
downloaded.

Run `make audit-manifest` to execute the same match/clip integrity check used by
manifest loading. A successful audit of an empty fixture proves only that the
schema and audit execute; it is not evidence that a dataset has been acquired.