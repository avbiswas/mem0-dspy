import asyncio
from mem.response_generator import run_chat
import sys

async def main(user_id):

    try:
        await run_chat(user_id)
    except KeyboardInterrupt:
        print("Exitting...")
        sys.exit(0)
    except asyncio.exceptions.CancelledError:
        sys.exit(0)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_id = int(sys.argv[1])
    else:
        user_id = 1
    response = asyncio.run(main(user_id))
