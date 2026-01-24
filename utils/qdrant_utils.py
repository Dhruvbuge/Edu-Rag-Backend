# utils/qdrant_utils.py

from qdrant_client import QdrantClient, models


def setup_qdrant(collection_name, vector_size, api_key, url):
    """
    Create/recreate Qdrant collection with given name and vector size.
    Uses a larger timeout to avoid write timeouts.
    """
    client = QdrantClient(
        url=url,
        api_key=api_key,
        timeout=60.0,  # increase if your network is slow
    )

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )
    return client


def upsert_chunks(qdrant_client, collection_name, chunks, embeddings, batch_size: int = 50):
    """
    Upload chunks + embeddings to Qdrant in small batches to avoid timeouts.
    """
    total = len(chunks)
    vectors = embeddings.tolist()  # convert numpy array to list

    print(f"💾 Upserting {total} chunks to Qdrant in batches of {batch_size}...")

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        batch_ids = list(range(start, end))
        batch_vectors = vectors[start:end]
        batch_payloads = [{"text": chunk} for chunk in chunks[start:end]]

        qdrant_client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=batch_ids,
                vectors=batch_vectors,
                payloads=batch_payloads,
            ),
            wait=True,
        )

        print(f"   ✅ Upserted points {start}–{end - 1}")

        
def search_qdrant(
    qdrant_client,
    collection_name,
    query_embedding,
    top_k: int = 5,
):
    result = qdrant_client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k,
        with_payload=True,
    )

    hits = result.points
    if not hits:
        return ""

    return "\n---\n".join([p.payload["text"] for p in hits])
