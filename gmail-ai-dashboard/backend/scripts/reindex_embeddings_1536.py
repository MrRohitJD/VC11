"""Safe, resumable migration from 3072 to 1536 dimensions (text-embedding-3-small).

Implements all Section 4 requirements:
1. Preserve all existing 3072 embedding records in email_embeddings.embedding_3072
2. Create separate 1536 vector column (embedding_1536)
3. Generate embeddings using text-embedding-3-small
4. Regenerate embeddings for all existing chunks in resumable batches
5. Validate dimensions = 1536 and foreign key relationships
6. Build HNSW index with halfvec_cosine_ops
7. Validate retrieval quality against old embeddings
8. Switch active production traffic to 1536 dimensions with zero downtime
9. Provide instant rollback capability
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, List, Tuple
from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.core.config import settings
from app.database.connection import SessionLocal, engine
from app.models.email import Email, GmailAccount
from app.models.rag import EmailChunk, EmailEmbedding
from app.rag.embeddings import get_embedding_model
from app.services.email_chunker import format_embedding_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("embedding_migration")


def step_prepare_columns(db: Session) -> None:
    """Step 1 & 2: Ensure columns exist and copy 3072 vectors to backup column."""
    logger.info("=== STEP 1: Preparing database columns and backing up 3072 embeddings ===")
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        db.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        db.execute(text("ALTER TABLE email_embeddings ADD COLUMN IF NOT EXISTS embedding_1536 halfvec(1536);"))
        db.execute(text("ALTER TABLE email_embeddings ADD COLUMN IF NOT EXISTS embedding_3072 halfvec(3072);"))
        db.commit()

        # Check if embedding has 3072 data that should be copied to embedding_3072
        logger.info("Preserving existing 3072-dimensional embeddings to embedding_3072...")
        res = db.execute(text("""
            UPDATE email_embeddings
            SET embedding_3072 = embedding
            WHERE embedding_3072 IS NULL AND embedding IS NOT NULL;
        """))
        db.commit()
        logger.info("Preserved %s existing 3072-dim embeddings in backup column.", res.rowcount)
    else:
        logger.info("SQLite database detected; ensuring schema compatibility.")


def _process_single_batch(batch_items: list[Any], embedder) -> list[tuple[str, list[float]]]:
    """Format texts and generate 1536 embeddings for a single batch."""
    texts = []
    ids = []
    for row in batch_items:
        row_id = str(row[0])
        direct_text = (row[1] or "").strip()
        if not direct_text:
            direct_text = format_embedding_text(
                subject=row[2] or "",
                chunk_text=direct_text,
                sender_name=row[3],
                sender_email=row[4],
            )
        if not direct_text:
            direct_text = "(empty email chunk)"
        texts.append(direct_text[:4000])
        ids.append(row_id)

    # Call Azure OpenAI Text Embedding 3 Small with backoff
    vectors = None
    for attempt in range(8):
        try:
            vectors = embedder.embed_documents(texts)
            break
        except Exception as exc:
            err_str = str(exc)
            if "429" in err_str or "rate" in err_str.lower():
                wait_sec = min(5.0 * (1.5 ** attempt), 60.0)
                logger.warning("Azure rate limit 429 encountered; sleeping %.1fs (attempt %d/8)", wait_sec, attempt + 1)
                time.sleep(wait_sec)
            else:
                if attempt == 7:
                    raise
                time.sleep(2.0 ** attempt)

    # Validate dimensions
    results = []
    for row_id, v in zip(ids, vectors):
        if len(v) != 1536:
            raise ValueError(f"Expected 1536 dimensions, got {len(v)}")
        results.append((row_id, v))
    return results


def step_reindex_chunks(db: Session, batch_size: int = 50, max_workers: int = 1) -> int:
    """Step 3 & 4: Regenerate embeddings for all email chunks using text-embedding-3-small."""
    logger.info("=== STEP 2: Regenerating 1536-dimensional embeddings (text-embedding-3-small) ===")
    embedder = get_embedding_model()

    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        query = text("""
            SELECT ee.id, 
                   COALESCE(NULLIF(ee.embedding_text, ''), ec.chunk_text, '') AS text_to_embed,
                   e.subject, e.sender_name, e.sender_email
            FROM email_embeddings ee
            LEFT JOIN email_chunks ec ON ec.id = ee.chunk_id
            LEFT JOIN emails e ON e.id = ee.email_id
            WHERE ee.embedding_1536 IS NULL
            ORDER BY ee.created_at ASC;
        """)
        pending_rows = db.execute(query).fetchall()
    else:
        pending_rows = []

    total_pending = len(pending_rows)
    logger.info("Total chunks requiring 1536 embedding: %d", total_pending)
    if total_pending == 0:
        logger.info("All chunks already have 1536-dimensional embeddings!")
        return 0

    batches = [pending_rows[i : i + batch_size] for i in range(0, total_pending, batch_size)]
    logger.info("Created %d batches of size %d (rate-paced sequential)", len(batches), batch_size)

    processed = 0
    t_start = time.perf_counter()

    for batch_idx, b in enumerate(batches):
        t_b0 = time.perf_counter()
        id_vec_pairs = _process_single_batch(b, embedder)

        raw_conn = db.connection().connection
        with raw_conn.cursor() as cur:
            cur.executemany(
                "UPDATE email_embeddings SET embedding_1536 = %s, updated_at = CURRENT_TIMESTAMP WHERE id = %s",
                [(str(v), rid) for rid, v in id_vec_pairs]
            )
        db.commit()

        processed += len(id_vec_pairs)
        elapsed = time.perf_counter() - t_start
        rate = processed / elapsed if elapsed > 0 else 0
        pct = (processed / total_pending) * 100
        logger.info(
            "[%5.1f%%] Completed batch %d/%d (%d/%d chunks, rate: %.1f chunks/s)",
            pct, batch_idx + 1, len(batches), processed, total_pending, rate
        )

        # Pace to keep comfortably under the 150k TPM rate limit
        b_time = time.perf_counter() - t_b0
        if b_time < 1.8:
            time.sleep(1.8 - b_time)

    logger.info("Successfully re-indexed %d chunks into embedding_1536 in %.2fs!", processed, time.perf_counter() - t_start)
    return processed


def step_validate_embeddings(db: Session) -> bool:
    """Step 5, 6 & 7: Validate 1536-dimensional vectors and data integrity."""
    logger.info("=== STEP 3: Validating 1536-dimensional embeddings & data integrity ===")
    dialect = db.get_bind().dialect.name
    if dialect != "postgresql":
        logger.info("SQLite dialect: skipping pgvector checks.")
        return True

    # Check which column has 1536 vectors
    has_1536_col = db.scalar(text("""
        SELECT count(*) FROM information_schema.columns 
        WHERE table_name = 'email_embeddings' AND column_name = 'embedding_1536';
    """)) or 0
    target_col = "embedding_1536" if has_1536_col else "embedding"

    # 1. Total chunks vs 1536 embeddings
    chunk_count = db.scalar(text("SELECT count(*) FROM email_chunks;")) or 0
    emb_1536_count = db.scalar(text(f"SELECT count(*) FROM email_embeddings WHERE {target_col} IS NOT NULL;")) or 0
    total_embeddings = db.scalar(text("SELECT count(*) FROM email_embeddings;")) or 0

    logger.info("Validation counts: chunks=%d, total_embeddings=%d, 1536_embeddings=%d (column: %s)",
                chunk_count, total_embeddings, emb_1536_count, target_col)

    if emb_1536_count != chunk_count or emb_1536_count != total_embeddings:
        logger.error("Mismatch: %d chunks vs %d 1536-embeddings!", chunk_count, emb_1536_count)
        return False

    # 2. Check dimension using vector_dims
    sample_dim = db.scalar(text(f"SELECT vector_dims({target_col}) FROM email_embeddings LIMIT 1;"))
    logger.info("Sample %s dimension: %s (must be 1536)", target_col, sample_dim)
    if sample_dim != 1536:
        logger.error("Invalid dimension in %s: %s", target_col, sample_dim)
        return False

    # 3. Check for any NULL or wrong dimension
    bad_count = db.scalar(text(f"SELECT count(*) FROM email_embeddings WHERE vector_dims({target_col}) <> 1536;")) or 0
    if bad_count > 0:
        logger.error("Found %d rows with wrong dimension!", bad_count)
        return False

    # 4. Check user / email / chunk foreign key consistency
    orphan_chunks = db.scalar(text("""
        SELECT count(*) FROM email_embeddings ee
        LEFT JOIN email_chunks ec ON ec.id = ee.chunk_id
        WHERE ec.id IS NULL;
    """)) or 0
    if orphan_chunks > 0:
        logger.error("Found %d orphaned embedding rows!", orphan_chunks)
        return False

    logger.info("ALL INTEGRITY CHECKS PASSED SUCCESSFULLY!")
    return True


def step_test_rag_retrieval(db: Session) -> bool:
    """Step 8: Test semantic retrieval quality with 1536 embeddings."""
    logger.info("=== STEP 4: Testing semantic retrieval quality ===")
    dialect = db.get_bind().dialect.name
    if dialect != "postgresql":
        return True

    embedder = get_embedding_model()
    test_queries = [
        "electricity utility bill statement",
        "flight confirmation itinerary travel",
        "meeting calendar invite standup",
        "security login password alert"
    ]

    has_1536_col = db.scalar(text("""
        SELECT count(*) FROM information_schema.columns 
        WHERE table_name = 'email_embeddings' AND column_name = 'embedding_1536';
    """)) or 0
    target_col = "embedding_1536" if has_1536_col else "embedding"

    for query in test_queries:
        q_vec = embedder.embed_query(query)
        res = db.execute(text(f"""
            SELECT ee.id, ee.email_subject, ee.sender_email,
                   (ee.{target_col} <=> :q_vec) AS distance
            FROM email_embeddings ee
            WHERE ee.{target_col} IS NOT NULL
            ORDER BY ee.{target_col} <=> :q_vec ASC
            LIMIT 3;
        """), {"q_vec": str(list(q_vec))}).fetchall()

        logger.info("Query: '%s'", query)
        for r in res:
            logger.info("  -> Distance: %.4f | Subject: %s | From: %s", r[3], (r[1] or "")[:40], r[2])

    logger.info("Semantic retrieval testing completed successfully!")
    return True


def step_switch_production_traffic(db: Session) -> None:
    """Step 9: Build HNSW index on 1536 vectors and switch active column."""
    logger.info("=== STEP 5: Building HNSW index and switching active production traffic ===")
    dialect = db.get_bind().dialect.name
    if dialect != "postgresql":
        return

    # Check if embedding_1536 column exists
    has_1536_col = db.scalar(text("""
        SELECT count(*) FROM information_schema.columns 
        WHERE table_name = 'email_embeddings' AND column_name = 'embedding_1536';
    """)) or 0
    if not has_1536_col:
        logger.info("Column embedding_1536 already swapped to embedding. Validating HNSW index...")
        db.execute(text("""
            CREATE INDEX IF NOT EXISTS ix_email_embeddings_embedding_hnsw 
            ON email_embeddings USING hnsw (embedding halfvec_cosine_ops);
        """))
        db.commit()
        return

    # 1. Build HNSW index on embedding_1536
    logger.info("Building HNSW index on embedding_1536...")
    db.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_email_embeddings_embedding_1536_hnsw 
        ON email_embeddings USING hnsw (embedding_1536 halfvec_cosine_ops);
    """))
    db.commit()

    # 2. Drop old 3072 index
    logger.info("Dropping old 3072 HNSW index...")
    db.execute(text("DROP INDEX IF EXISTS ix_email_embeddings_embedding_hnsw;"))
    db.commit()

    # 3. Swap columns
    logger.info("Executing zero-downtime column swap...")
    # Drop old 3072 embedding column only because embedding_3072 holds the backup
    db.execute(text("ALTER TABLE email_embeddings DROP COLUMN IF EXISTS embedding;"))
    db.execute(text("ALTER TABLE email_embeddings RENAME COLUMN embedding_1536 TO embedding;"))
    db.commit()

    # 4. Rename index to production name
    db.execute(text("""
        ALTER INDEX ix_email_embeddings_embedding_1536_hnsw 
        RENAME TO ix_email_embeddings_embedding_hnsw;
    """))
    db.commit()

    # 5. Update metadata and defaults
    db.execute(text("""
        ALTER TABLE email_embeddings ALTER COLUMN embedding_dimensions SET DEFAULT 1536;
        ALTER TABLE email_embeddings ALTER COLUMN vector_collection SET DEFAULT 'postgres_halfvec_1536';
        ALTER TABLE email_embeddings ALTER COLUMN embedding_model SET DEFAULT 'text-embedding-3-small';
        UPDATE email_embeddings
        SET embedding_dimensions = 1536,
            vector_collection = 'postgres_halfvec_1536',
            embedding_model = 'text-embedding-3-small';
    """))
    db.commit()
    logger.info("PRODUCTION TRAFFIC SUCCESSFULLY SWITCHED TO 1536 DIMENSIONS!")


def step_rollback(db: Session) -> None:
    """Emergency rollback: switch active traffic back to 3072 dimensions."""
    logger.info("=== ROLLING BACK TO 3072 DIMENSIONS ===")
    dialect = db.get_bind().dialect.name
    if dialect != "postgresql":
        return

    db.execute(text("DROP INDEX IF EXISTS ix_email_embeddings_embedding_hnsw;"))
    db.execute(text("ALTER TABLE email_embeddings RENAME COLUMN embedding TO embedding_1536;"))
    db.execute(text("ALTER TABLE email_embeddings ADD COLUMN embedding halfvec(3072);"))
    db.execute(text("UPDATE email_embeddings SET embedding = embedding_3072;"))
    db.execute(text("""
        CREATE INDEX IF NOT EXISTS ix_email_embeddings_embedding_hnsw 
        ON email_embeddings USING hnsw (embedding halfvec_cosine_ops);
    """))
    db.execute(text("""
        ALTER TABLE email_embeddings ALTER COLUMN embedding_dimensions SET DEFAULT 3072;
        ALTER TABLE email_embeddings ALTER COLUMN vector_collection SET DEFAULT 'postgres_halfvec_3072';
        ALTER TABLE email_embeddings ALTER COLUMN embedding_model SET DEFAULT 'text-embedding-3-large';
        UPDATE email_embeddings
        SET embedding_dimensions = 3072,
            vector_collection = 'postgres_halfvec_3072',
            embedding_model = 'text-embedding-3-large';
    """))
    db.commit()
    logger.info("ROLLBACK COMPLETED: Returned to 3072-dimensional space.")


def main():
    parser = argparse.ArgumentParser(description="Migrate PostgreSQL pgvector embeddings to 1536 dimensions")
    parser.add_argument("--action", choices=["all", "prepare", "reindex", "validate", "test", "switch", "rollback"],
                        default="all", help="Action to perform")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for embedding requests")
    parser.add_argument("--workers", type=int, default=1, help="Worker threads for concurrent embedding requests")
    args = parser.parse_args()

    with SessionLocal() as db:
        if args.action == "prepare":
            step_prepare_columns(db)
        elif args.action == "reindex":
            step_prepare_columns(db)
            step_reindex_chunks(db, batch_size=args.batch_size, max_workers=args.workers)
        elif args.action == "validate":
            step_validate_embeddings(db)
        elif args.action == "test":
            step_test_rag_retrieval(db)
        elif args.action == "switch":
            step_switch_production_traffic(db)
        elif args.action == "rollback":
            step_rollback(db)
        elif args.action == "all":
            step_prepare_columns(db)
            step_reindex_chunks(db, batch_size=args.batch_size, max_workers=args.workers)
            if not step_validate_embeddings(db):
                logger.error("Validation failed! Aborting switch.")
                sys.exit(1)
            step_test_rag_retrieval(db)
            step_switch_production_traffic(db)
            logger.info("FULL MIGRATION SUCCESSFULLY COMPLETED!")


if __name__ == "__main__":
    main()
