#!/usr/bin/env python3
"""Ingest Oracle HCM schema metadata into a Pinecone index."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict
from typing import Iterable
from typing import List

from pinecone import Pinecone
from pinecone import ServerlessSpec
from sentence_transformers import SentenceTransformer

from env_config import load_env_file


load_env_file()


PINECONE_API_KEY = os.getenv(
    "PINECONE_API_KEY",
)
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "chatbot")
JSON_PATH = Path(os.getenv("HCM_SCHEMA_JSON_PATH", "hcm_structured.json"))
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"
BATCH_SIZE = 100
READY_TIMEOUT_SECONDS = 300
READY_POLL_INTERVAL_SECONDS = 5
UPSERT_MAX_RETRIES = 5
UPSERT_RETRY_BASE_SECONDS = 2


logger = logging.getLogger("ingest_hcm_schema")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_schema(json_path: Path) -> Dict:
    if not json_path.exists():
        raise FileNotFoundError(f"Schema file not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def normalize_category(default_category: str, raw_category: str | None) -> str:
    value = (raw_category or "").strip().upper()
    if value.startswith("TABLE"):
        return "TABLE"
    if value.startswith("VIEW"):
        return "VIEW"
    return default_category


def extract_column_names(columns: List[Dict]) -> List[str]:
    return [str(column.get("name", "")).strip() for column in columns if column.get("name")]


def detect_effective_dated(column_names: List[str]) -> bool:
    normalized = {name.upper() for name in column_names}
    return "EFFECTIVE_START_DATE" in normalized or "EFFECTIVE_DATE" in normalized


def build_semantic_summary(
    module_name: str,
    name: str,
    category: str,
    description: str,
    column_names: List[str],
) -> str:
    description_text = description.strip() or "No description provided."
    joined_columns = ", ".join(column_names) if column_names else "No columns listed"
    return (
        f"Module: {module_name}. "
        f"Name: {name}. "
        f"Type: {category}. "
        f"Description: {description_text}. "
        f"Columns: {joined_columns}."
    )


def iter_vector_records(schema: Dict) -> Iterable[Dict]:
    modules = schema.get("modules", [])
    for module in modules:
        module_name = str(module.get("module_name") or module.get("module") or "Unknown").strip()

        for object_group, default_category in (("tables", "TABLE"), ("views", "VIEW")):
            for obj in module.get(object_group, []):
                name = str(obj.get("name", "")).strip()
                if not name:
                    continue

                columns = obj.get("columns", [])
                column_names = extract_column_names(columns)
                category = normalize_category(default_category, obj.get("category"))
                primary_key_columns = obj.get("primary_key", {}).get("columns", []) or []
                primary_key = str(primary_key_columns[0]).strip() if primary_key_columns else ""
                description = str(obj.get("description", "") or "")

                yield {
                    "id": name,
                    "text": build_semantic_summary(
                        module_name=module_name,
                        name=name,
                        category=category,
                        description=description,
                        column_names=column_names,
                    ),
                    "metadata": {
                        "table_name": name,
                        "module": module_name,
                        "category": category,
                        "primary_key": primary_key,
                        "is_effective_dated": detect_effective_dated(column_names),
                    },
                }


def ensure_index(pc: Pinecone, index_name: str) -> None:
    existing_indexes = pc.list_indexes().names()
    if index_name not in existing_indexes:
        logger.info(
            "Creating Pinecone index '%s' with %s dimensions and %s metric",
            index_name,
            EMBEDDING_DIMENSION,
            METRIC,
        )
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric=METRIC,
            spec=ServerlessSpec(cloud=CLOUD, region=REGION),
        )
    else:
        description = pc.describe_index(index_name)
        dimension = getattr(description, "dimension", None)
        metric = getattr(description, "metric", None)
        if dimension != EMBEDDING_DIMENSION or str(metric).lower() != METRIC:
            raise ValueError(
                f"Index '{index_name}' exists with dimension={dimension}, metric={metric}. "
                f"Expected dimension={EMBEDDING_DIMENSION}, metric={METRIC}."
            )

    wait_for_index_ready(pc, index_name)


def wait_for_index_ready(pc: Pinecone, index_name: str) -> None:
    logger.info("Waiting for Pinecone index '%s' to become ready", index_name)
    deadline = time.time() + READY_TIMEOUT_SECONDS

    while time.time() < deadline:
        description = pc.describe_index(index_name)
        status = getattr(description, "status", {}) or {}
        ready = False

        if isinstance(status, dict):
            ready = bool(status.get("ready"))
        else:
            ready = bool(getattr(status, "ready", False))

        if ready:
            logger.info("Index '%s' is ready", index_name)
            return

        time.sleep(READY_POLL_INTERVAL_SECONDS)

    raise TimeoutError(
        f"Timed out after {READY_TIMEOUT_SECONDS} seconds waiting for index '{index_name}' to become ready."
    )


def chunked(items: List[Dict], batch_size: int) -> Iterable[List[Dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def encode_records(model: SentenceTransformer, records: List[Dict]) -> List[Dict]:
    texts = [record["text"] for record in records]
    embeddings = model.encode(
        texts,
        batch_size=min(BATCH_SIZE, len(texts)),
        show_progress_bar=False,
        normalize_embeddings=True,
    )

    vectors = []
    for record, embedding in zip(records, embeddings):
        vectors.append(
            {
                "id": record["id"],
                "values": embedding.tolist(),
                "metadata": record["metadata"],
            }
        )
    return vectors


def upsert_with_retry(index, vectors: List[Dict]) -> None:
    for attempt in range(1, UPSERT_MAX_RETRIES + 1):
        try:
            index.upsert(vectors=vectors)
            return
        except Exception as exc:
            message = str(exc).lower()
            is_timeout = "timeout" in message or "timed out" in message
            if attempt == UPSERT_MAX_RETRIES or not is_timeout:
                raise

            sleep_seconds = UPSERT_RETRY_BASE_SECONDS ** attempt
            logger.warning(
                "Upsert attempt %s/%s timed out. Retrying in %s seconds.",
                attempt,
                UPSERT_MAX_RETRIES,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)


def main() -> int:
    configure_logging()

    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is required.")

    logger.info("Loading schema from %s", JSON_PATH)
    schema = load_schema(JSON_PATH)
    records = list(iter_vector_records(schema))
    if not records:
        logger.warning("No tables or views were found in %s", JSON_PATH)
        return 0

    logger.info("Prepared %s table/view records for ingestion", len(records))
    logger.info("Loading local embedding model %s", EMBEDDING_MODEL_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    pc = Pinecone(api_key=PINECONE_API_KEY)
    ensure_index(pc, INDEX_NAME)
    index = pc.Index(INDEX_NAME)

    total_batches = (len(records) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_number, batch in enumerate(chunked(records, BATCH_SIZE), start=1):
        vectors = encode_records(model, batch)
        upsert_with_retry(index, vectors)
        logger.info(
            "Upserted batch %s/%s (%s vectors)",
            batch_number,
            total_batches,
            len(vectors),
        )

    logger.info("Finished ingesting %s records into Pinecone index '%s'", len(records), INDEX_NAME)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        logger.exception("Ingestion failed: %s", exc)
        raise SystemExit(1) from exc
