from datetime import datetime
from typing import Optional
from uuid import uuid4
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.grpc import Points, ScoredPoint
from qdrant_client.models import Distance, Filter, VectorParams, models
import asyncio
import numpy as np

client = AsyncQdrantClient(url="http://localhost:6333")
COLLECTION_NAME = "memories"


class EmbeddedMemory(BaseModel):
    user_id: int
    memory_text: str
    categories: list[str]
    date: str
    embedding: list[float]


class RetrievedMemory(BaseModel):
    point_id: str
    user_id: int
    memory_text: str
    categories: list[str]
    date: str
    score: float


async def create_memory_collection():
    if not (await client.collection_exists(COLLECTION_NAME)):
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=64, distance=Distance.DOT),
        )

        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="user_id",
            field_schema=models.PayloadSchemaType.INTEGER,
        )

        # Create an index on the 'categories' field
        await client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="categories",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print("Collection created")
    else:
        print("Collection exists")


async def insert_memories(memories: list[EmbeddedMemory]):
    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=uuid4().hex,
                payload={
                    "user_id": memory.user_id,
                    "categories": memory.categories,
                    "memory_text": memory.memory_text,
                    "date": memory.date,
                },
                vector=memory.embedding,
            )
            for memory in memories
        ],
    )


async def search_memories(
    search_vector: list[float],
    user_id: int,
    categories: Optional[list[str]] = None,
):
    must_conditions: list[models.Condition] = [
        models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
    ]
    if categories is not None:
        if len(categories) > 0:
            must_conditions.append(
                models.FieldCondition(
                    key="categories", match=models.MatchAny(any=categories)
                )
            )
    outs = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=search_vector,
        with_payload=True,
        query_filter=Filter(must=must_conditions),
        score_threshold=0.1,
        limit=2,
    )

    return [
        convert_retrieved_records(point) for point in outs.points if point is not None
    ]


async def delete_user_records(user_id):
    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="user_id", match=models.MatchValue(value=user_id)
                    )
                ]
            )
        ),
    )


async def delete_records(point_ids):
    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.PointIdsList(points=point_ids),
    )


async def fetch_all_user_records(user_id):
    out = await client.query_points(
        collection_name=COLLECTION_NAME,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="user_id", match=models.MatchValue(value=user_id)
                )
            ]
        ),
    )
    return [convert_retrieved_records(point) for point in out.points]


def convert_retrieved_records(point) -> RetrievedMemory:
    return RetrievedMemory(
        point_id=point.id,
        user_id=point.payload["user_id"],
        memory_text=point.payload["memory_text"],
        categories=point.payload["categories"],
        date=point.payload["date"],
        score=point.score,
    )


async def get_all_categories(user_id):
    """
    Uses Qdrant's facet feature to efficiently get all unique categories
    from the indexed 'categories' field.
    """
    facet_filter = Filter(
        must=[
            models.FieldCondition(key="user_id", match=models.MatchValue(value=user_id))
        ]
    )

    # Use the facet method to get unique values from the indexed field
    facet_result = await client.facet(
        collection_name=COLLECTION_NAME,
        key="categories",
        facet_filter=facet_filter,
        limit=1000,  # Maximum number of unique categories to return
    )

    # Use the facet method to get unique values from the indexed field
    facet_result = await client.facet(
        collection_name=COLLECTION_NAME,
        key="categories",
        facet_filter=facet_filter,
        limit=1000,  # Maximum number of unique categories to return
    )

    unique_categories = [hit.value for hit in facet_result.hits]

    return unique_categories


def stringify_retrieved_point(retrieved_memory: RetrievedMemory):
    return f"""{retrieved_memory.memory_text} (Categories: {retrieved_memory.categories}) Relevance: {retrieved_memory.score:.2f}"""


if __name__ == "__main__":
    asyncio.run(create_memory_collection())
