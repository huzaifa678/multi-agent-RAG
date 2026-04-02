import asyncio

def sync_call(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return asyncio.create_task(coro)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)