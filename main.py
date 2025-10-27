import asyncio
from mem.response_generator import run_chat

async def main():
    import sys

    user_id = 1
    try:
        await run_chat(user_id)
    except KeyboardInterrupt:
        print("Exitting...")
        sys.exit(0)
    except asyncio.exceptions.CancelledError:
        sys.exit(0)


if __name__ == "__main__":
    response = asyncio.run(main())
