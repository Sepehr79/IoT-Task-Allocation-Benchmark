import asyncio
import ssl
import time

import websockets


URI = "wss://localhost:8765"


async def benchmark():

    ssl_context = ssl.create_default_context(
        cafile="server.crt"
    )

    results = []

    for i in range(20):

        start_time = time.perf_counter()

        async with websockets.connect(
            URI,
            ssl=ssl_context
        ) as websocket:

            end_time = time.perf_counter()

            connection_delay = (
                end_time - start_time
            ) * 1000

            results.append(connection_delay)

            print(
                f"Trial {i + 1:02d}: "
                f"{connection_delay:.3f} ms"
            )

    print("\nResults")
    print(f"Minimum : {min(results):.3f} ms")
    print(f"Maximum : {max(results):.3f} ms")
    print(f"Average : {sum(results) / len(results):.3f} ms")


asyncio.run(benchmark())