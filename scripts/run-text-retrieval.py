"""Run text retrieval evals on the GDZ dataset via Weaviate Cloud.

Uploads the GDZ corpus to Weaviate, then evaluates BM25, vector, and
hybrid search using the query-agent-benchmarking library.
Queries are passed in-memory to avoid needing a second collection.
"""

import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
from weaviate.client import WeaviateClient
from weaviate.connect import ConnectionParams, ProtocolParams
from datasets import load_dataset

from query_agent_benchmarking import (
    DocsCollection,
    InMemoryQuery,
    run_search_eval,
)

_env_path = Path(__file__).resolve().parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

WEAVIATE_HOST = os.getenv(
    "WEAVIATE_HOST",
    os.getenv("WEAVIATE_URL", "ihdoqiiwsc8hc2kldbpua.c0.eu-central-1.aws.weaviate.cloud"),
)
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
DOCS_COLLECTION = "GDZ_Default"


def load_gdz_dataset():
    raw_docs = load_dataset(
        "Trungdaik/Visual_information_retrieval", "docs", split="train"
    )
    raw_queries = load_dataset(
        "Trungdaik/Visual_information_retrieval", "queries", split="train"
    )
    docs = [dict(item) for item in raw_docs]
    queries = [
        {
            "question": item["question"],
            "query_id": str(item["dataset_id"]),
            "dataset_ids": [str(item["dataset_id"])],
        }
        for item in raw_queries
    ]
    return docs, queries


def connect_to_weaviate():
    cluster_url = WEAVIATE_HOST
    if cluster_url.startswith("http"):
        cluster_url = urlparse(cluster_url).netloc
    if cluster_url.endswith(".weaviate.network"):
        ident, domain = cluster_url.split(".", 1)
        grpc_host = f"{ident}.grpc.{domain}"
    else:
        grpc_host = f"grpc-{cluster_url}"
    client = WeaviateClient(
        connection_params=ConnectionParams(
            http=ProtocolParams(host=cluster_url, port=443, secure=True),
            grpc=ProtocolParams(host=grpc_host, port=443, secure=True),
        ),
        auth_client_secret=Auth.api_key(WEAVIATE_API_KEY),
        skip_init_checks=True,
    )
    client.connect()
    return client


def ensure_docs_collection(client):
    if client.collections.exists(DOCS_COLLECTION):
        client.collections.delete(DOCS_COLLECTION)

    client.collections.create(
        name=DOCS_COLLECTION,
        vector_config=Configure.Vectors.text2vec_weaviate(),
        properties=[
            Property(
                name="dataset_id", data_type=DataType.TEXT, skip_vectorization=True
            ),
            Property(name="transcription", data_type=DataType.TEXT),
        ],
    )


def upload_docs(client, docs):
    collection = client.collections.get(DOCS_COLLECTION)
    batch_size = 50
    for start in range(0, len(docs), batch_size):
        with collection.batch.dynamic() as batch:
            for doc in docs[start:start + batch_size]:
                batch.add_object(
                    properties={
                        "dataset_id": str(doc.get("dataset_id", "")),
                        "transcription": str(doc.get("transcription", "")),
                    }
                )
        print(f"  Uploaded batch {start // batch_size + 1}/{(len(docs) + batch_size - 1) // batch_size}")


def main():
    print("Loading GDZ dataset...")
    docs, queries = load_gdz_dataset()
    print(f"Loaded {len(docs)} docs, {len(queries)} queries")

    print("Connecting to Weaviate...")
    client = connect_to_weaviate()

    try:
        print("Creating/recreating docs collection...")
        ensure_docs_collection(client)

        print("Uploading documents...")
        upload_docs(client, docs)
        print("Documents uploaded.")
    finally:
        client.close()

    docs_collection = DocsCollection(
        collection_name=DOCS_COLLECTION,
        content_key="transcription",
        id_key="dataset_id",
    )
    in_memory_queries = [
        InMemoryQuery(
            question=q["question"],
            dataset_ids=q["dataset_ids"],
            query_id=q["query_id"],
        )
        for q in queries
    ]

    for agent_name in ["bm25-search", "vector-search", "hybrid-search"]:
        print(f"\nRunning {agent_name} eval...")
        run_search_eval(
            docs_collection=docs_collection,
            queries=in_memory_queries,
            agent_name=agent_name,
            search_agent_name=agent_name,
            num_trials=1,
            use_async=False,
            output_path=f"console/results/gdz-{agent_name}",
        )
        print(f"Finished {agent_name} eval.")


if __name__ == "__main__":
    main()
