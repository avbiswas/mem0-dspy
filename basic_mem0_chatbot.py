import openai
from mem0 import MemoryClient

user_id = "avb"

memory = MemoryClient()
client = openai.Client()

messages = []


while True:
    user_input = input("User: ")

    messages.append(
        {
            "role": "user", "content": user_input
        }
    )

    related_memories = memory.search(user_input, user_id=user_id)
    print(related_memories)


    related_memories_text = "/n -".join([
        f"{m['memory']}"
        for m in related_memories
    ])

    print(related_memories_text)

    system_message = [
        {
            "role": "system",
            "content": f"""answer the user's question honestly.
Here are some relevant information you may find useful that previous interactions with the user has taught us:
{related_memories_text}
        """
        }
    ]
    
    

    response = client.chat.completions.create(
        messages=system_message + messages,
        model="gpt-5-mini",
        reasoning_effort="minimal"
    )

    answer = response.choices[0].message.content

    messages.append(
        {
            "role": "assistant", "content": answer
        }
    )
    print(f"\nAssistant: {answer} \n")

    memory.add(messages[-2:], user_id=user_id)



