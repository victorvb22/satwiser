# Backend (step 5)

Read-only FastAPI service. Two data sources, selected with `SATWISER_DATA_BACKEND`:

- `local` — reads the aggregates written by the pipeline under `$SATWISER_DATA_DIR/processed`
  (local development and preview, no account needed);
- `supabase` — reads the same tables from Supabase Postgres (hosted deployment on Render).
