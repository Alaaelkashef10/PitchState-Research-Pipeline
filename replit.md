# PitchState Research Pipeline

PitchState is a research-engineering foundation for confidence-aware pitch-space reconstruction and team-shape measurement from broadcast football video.

## Run & Operate

- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages
- `pnpm --filter @workspace/api-spec run codegen` — regenerate API hooks and Zod schemas from the OpenAPI spec
- `pnpm --filter @workspace/db run push` — push DB schema changes (dev only)
- Required env: `DATABASE_URL` — Postgres connection string
- `cd research/pitchstate && make smoke` — run the deterministic Phase 0 smoke pipeline
- `cd research/pitchstate && make test` — run the Phase 0 Python test suite

## Stack

- pnpm workspaces, Node.js 24, TypeScript 5.9
- API: Express 5
- DB: PostgreSQL + Drizzle ORM
- Validation: Zod (`zod/v4`), `drizzle-zod`
- API codegen: Orval (from OpenAPI spec)
- Build: esbuild (CJS bundle)

## Where things live

- `research/pitchstate/src/pitchstate/` — modular Python research package
- `research/pitchstate/configs/` — versioned TOML experiment configurations
- `research/pitchstate/data/manifests/` — dataset/version manifests; raw data is never committed
- `research/pitchstate/docs/research-plan.md` — source of truth for the narrowed research question and scope
- `research/pitchstate/docs/architecture.md` — stage contracts and decision register
- `research/pitchstate/tests/` — unit and smoke tests

## Architecture decisions

- The first research claim is confidence-aware pitch-space team-shape measurement, not a new detector.
- Python is the ML/research runtime; Phase 0 uses only the standard library so setup is reproducible without GPU packages.
- Detection, tracking, calibration, and tactical analysis are dependency-injected protocols, allowing model replacement without changing orchestration.
- Camera cuts invalidate shot-local state; the first version abstains instead of attempting cross-cut identity continuity.
- Outputs carry confidence and abstention reasons so downstream analytics cannot silently turn missing evidence into facts.

## Product

The current deliverable is a research repository, not a user-facing application. It can validate configuration, dataset manifests, experiment metadata, and a synthetic end-to-end pipeline. Later phases will add real video inference, evaluation, and visual reporting.

## User preferences

- Prefer scientific honesty, explicit uncertainty, reproducible experiments, and simpler baselines over impressive but unvalidated tactical outputs.

## Gotchas

- Do not download or commit benchmark videos into the repository; use manifests and local paths.
- The Phase 0 smoke test is deliberately synthetic and is not evidence of model or research performance.
- Do not add a tactical feature without a defined reference annotation, metric, and abstention policy.

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
