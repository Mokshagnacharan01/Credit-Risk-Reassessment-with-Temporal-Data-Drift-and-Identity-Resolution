# Real-Time Credit Risk Reassessment

A system that ingests time-stamped, asynchronous, and potentially conflicting
customer financial events, resolves customer identity across them, detects
temporal drift in feature distributions, and maintains a versioned,
auditable, idempotent risk score per customer.

## What's inside

| Module | Responsibility |
|---|---|
| src/identity_resolution.py | Deterministic identity resolution (account_id + fuzzy customer_id/transaction-history matching via Union-Find) |
| src/drift_detection.py | Rolling 30-day drift detection using a hand-rolled two-sample KS test (no SciPy, per constraints) + mean-shift check |
| src/risk_model.py | Logistic regression training/scoring, model versioning, persisted per-customer risk state |
| src/event_log.py | Idempotency ledger — tracks processed event_ids so replays never change state |
| src/audit.py | Writes a JSON audit trail for every decision (including skipped/duplicate ones) |
| src/event_processor.py | Orchestrates the pipeline end-to-end; the core EventProcessor class |
| src/cli.py | Command-line ingestion (bootstrap, ingest) |
| src/api.py | FastAPI server exposing POST /events |

## Clone → Setup → Run → Test
