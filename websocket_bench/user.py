import asyncio
import ssl
import time

from websockets.asyncio.client import connect


SERVER_IP = "192.168.246.250"
SERVER_PORT = 8765
URI = f"wss://{SERVER_IP}:{SERVER_PORT}"

NUM_TRIALS = 1


async def main():

    ssl_context = ssl.create_default_context(
        cafile="server.crt"
    )

    results = []

    conn = True
    if conn:
        conn = False

        for i in range(NUM_TRIALS):

            start = time.perf_counter()

            async with connect(
                URI,
                ssl=ssl_context
            ) as websocket:

                end = time.perf_counter()

            delay_ms = (end - start) * 1000
            results.append(delay_ms)

            print(
                f"Trial {i + 1:03d}: "
                f"{delay_ms:.3f} ms"
            )


            # Wait before next connection
            await asyncio.sleep(1)

        average = sum(results) / len(results)
        minimum = min(results)
        maximum = max(results)

        print("\nResults")
        print(f"Average: {average:.3f} ms")
        print(f"Minimum: {minimum:.3f} ms")
        print(f"Maximum: {maximum:.3f} ms")


if __name__ == "__main__":
    asyncio.run(main())