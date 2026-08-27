# Experiments

Each real experiment should have:

- An immutable configuration file.
- A dataset manifest and split identifier.
- A deterministic seed.
- A code revision identifier.
- Cached upstream predictions when appropriate.
- JSONL run metadata and machine-readable metrics.
- A short notes file describing the hypothesis, failure cases, and deviations.

Local run logs are written to `experiments/runs/` and are ignored by git.
Phase 0 uses the same logging contract but the smoke output is synthetic.