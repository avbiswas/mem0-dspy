import asyncio
import openai
import numpy as np

client = openai.AsyncClient()
async def generate_embeddings(strings: list[str]):
    out = await client.embeddings.create(
        input=strings,
        model="text-embedding-3-small",
        dimensions=64
    )
    embeddings = [item.embedding for item in out.data]
    return embeddings

if __name__ == "__main__":
    texts = [
        "Hello how are you",
        "I like Machine Learning"
    ]
    asyncio.run(generate_embeddings(texts))
