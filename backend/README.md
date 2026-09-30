# Backend

Read-only FastAPI service over the aggregates exported by `scripts/export_app_data.py`.

Local run (no account needed, reads `$SATWISER_DATA_DIR/app`):

    uvicorn satwiser_api.main:app --reload --app-dir backend

Interactive documentation: http://127.0.0.1:8000/docs

Hosted: `SATWISER_DATA_BACKEND=database` and `DATABASE_URL` (Supabase Postgres, set as a
Render secret). Load the database with `python backend/scripts/load_database.py`; the
tables are listed in `sql/schema.sql`.
