#!/bin/sh
set -eu
export LANGSMITH_TRACING=false LANGCHAIN_TRACING_V2=false

python scripts/reconcile_alembic.py
alembic upgrade head
python scripts/seed_demo_mailbox.py
