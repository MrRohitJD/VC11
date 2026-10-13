"""CLI script to migrate existing PostgreSQL email chunks to ChromaDB gmail_email_chunks_v2 using OpenAI text-embedding-3-large."""

import argparse
import os
import sys

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.database.connection import SessionLocal
from app.services.rag_migration_service import RAGMigrationService


def main():
    parser = argparse.ArgumentParser(description="Migrate email chunks to ChromaDB v2 collection with Azure OpenAI text-embedding-3-large")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for Azure OpenAI embedding requests")
    args = parser.parse_args()

    db = SessionLocal()
    try:
        service = RAGMigrationService()
        deployment = getattr(service.embedder, "deployment_name", "text-embedding-3-large")
        dims = getattr(service.embedder, "dimensions", 3072)
        provider = getattr(service.embedder, "provider", "azure_openai")

        print("================================================================")
        print("  GMAIL RAG: MIGRATING TO AZURE OPENAI text-embedding-3-large    ")
        print("================================================================")
        print(f"Provider:          {provider}")
        print(f"Azure Deployment:  {deployment}")
        print(f"Target Collection: {service.target_collection} ({dims} dimensions)")
        print(f"Batch Size:        {args.batch_size}")
        print()

        def on_progress(p):
            print(f"[{p.percent_complete:5.1f}%] Processed: {p.already_indexed + p.migrated}/{p.total_chunks} chunks (Migrated: {p.migrated}, Failed: {p.failed}) - {p.elapsed_seconds:.1f}s")

        summary = service.migrate_all_chunks(db=db, batch_size=args.batch_size, progress_callback=on_progress)
        print()
        print("Migration Completed Successfully!")
        print(f"Total Chunks: {summary.total_chunks}")
        print(f"Newly Migrated: {summary.migrated}")
        print(f"Previously Indexed: {summary.already_indexed}")
        print(f"Failed: {summary.failed}")
        print(f"Total Time: {summary.elapsed_seconds}s")
    finally:
        db.close()


if __name__ == "__main__":
    main()
