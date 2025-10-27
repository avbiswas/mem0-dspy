from typing import Dict, List
import dspy
import asyncio
from rich.console import Console
from rich.rule import Rule

import warnings

warnings.filterwarnings("ignore")

from mem.generate_embeddings import generate_embeddings
from mem.update_memory import update_memories
from mem.vectordb import get_all_categories, search_memories, stringify_retrieved_point

console = Console(log_path=False)

model = dspy.LM(model="gemini/gemini-2.0-flash")


class ResponseGenerator(dspy.Signature):
    """
    You will be given a past conversation transcript between user and an AI agent. Also the latest question by the user.

    You have the option to look up the past memories from a vector database to fetch relevant context if required. If you can't find the answer to user's question from transcript or from your own internal knowledge, use the provided search tool calls to search for information.

    You are also provided a list of existing categories in the memory database to quickly search across categories. You can select multiple categories as a list to do your searches. If you select no categories (keep it empty). If you keep categories as empty, we will simply search across the entire database - that is fine too.

    The retrieved information may or may not contain the information user wants. React accordingly.

    You must output the final response, and also decide the latest interaction needs to be recorded into the memory database. New memories are meant to store new information that the user provides.

    While responding, you must be aware that you are continuously learning new memories about the user, so if retrieved memories do not directly address the user's question, mention what you know, acknowledge the gaps in your knowledge, and ask the user for information.

    If you retrieved records using the search tools, and the information was already present, no need to save a new memory. Only save memory if the new information is richer than what you retrieved or didn't find.

    New memories should be made when the USER provides new info. It is not to save information about the the AI or the assistant.
    """

    transcript: list[dict] = dspy.InputField()
    existing_categories: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    response: str = dspy.OutputField()
    save_memory: bool = dspy.OutputField(
        description="True if a new memory record needs to be created for the latest interaction"
    )


async def run_chat(user_id):

    async def fetch_similar_memories(search_text: str, categories: list[str]):
        """
        Search memories from vector database if conversation requires additional context.

        Args:
        - search_text : The string to embed and do vector similarity search
        - categories : List of strings taken from existing_categories. Use an empty list ( [] ) if you want to search across all categories.
        """

        console.log("Search text: ", search_text)
        console.log("Categories: ", categories)

        search_vector = (await generate_embeddings([search_text]))[0]
        memories = await search_memories(
            search_vector,
            user_id=user_id,
            categories=None if len(categories) == 0 else categories,
        )
        memories_str = [stringify_retrieved_point(m_) for m_ in memories]
        console.log(f"Retrieved memories: \n", "\n- ".join(memories_str))
        return {"memories": memories_str}

    response_generator = dspy.ReAct(
        ResponseGenerator, tools=[fetch_similar_memories], max_iters=2
    )
    past_messages = []

    existing_categories = await get_all_categories(user_id=user_id)

    console.print("Let's begin to chat!", style="bold green")

    while True:
        question = console.input("[bold cyan]> [/bold cyan]")
        console.print(Rule(style="grey50"))

        with console.status("[bold green] Working..."):

            # We are passing the entire conversation stack of the current session to the model
            # Mem0 does not do this. It seperately creates a conversation summary when chats get too long.
            # I ignored this step to keep the code simple.
            with dspy.context(lm=model):
                out = await response_generator.acall(
                    transcript=past_messages,  # TODO: Unbounded transcript
                    question=question,
                    existing_categories=existing_categories,
                )

            response = out.response

            past_messages.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": response},
                ]
            )

            if out.save_memory:
                # Ideally, this should run as a background process
                # But I am running it here to showcase the workflow better

                console.log("Trying to update memory...")
                update_result = await update_memories(
                    user_id=user_id,
                    messages=past_messages[-6:],
                )
                console.log(update_result, style="italic")

                # Refresh the existing categories that's searchable
                existing_categories = await get_all_categories(user_id=user_id)

        console.print(f"\nAI: {response}\n\n", style="bold green")
