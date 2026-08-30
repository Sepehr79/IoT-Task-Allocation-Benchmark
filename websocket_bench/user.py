import asyncio
import ssl
import time

from websockets.asyncio.client import connect


SERVER_IP = "192.168.173.207"
SERVER_PORT = 8765
URI = f"wss://{SERVER_IP}:{SERVER_PORT}"

NUM_TRIALS = 20


async def main():

    ssl_context = ssl.create_default_context(
        cafile="server.crt"
    )

    results = []

    # Create file from scratch
    with open("connection_results_half_second.txt", "w") as f:

        f.write(f"Server: {SERVER_IP}\n")
        f.write(f"Port: {SERVER_PORT}\n")
        f.write("-" * 40 + "\n")

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

            f.write(
                f"Trial {i + 1:03d}: "
                f"{delay_ms:.3f} ms\n"
            )

            # Wait before next connection
            await asyncio.sleep(0.5)

        average = sum(results) / len(results)
        minimum = min(results)
        maximum = max(results)

        print("\nResults")
        print(f"Average: {average:.3f} ms")
        print(f"Minimum: {minimum:.3f} ms")
        print(f"Maximum: {maximum:.3f} ms")

        f.write("\nResults\n")
        f.write(f"Average: {average:.3f} ms\n")
        f.write(f"Minimum: {minimum:.3f} ms\n")
        f.write(f"Maximum: {maximum:.3f} ms\n")


if __name__ == "__main__":
    asyncio.run(main())