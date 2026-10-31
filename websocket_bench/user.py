import asyncio
import ssl
import time
from pathlib import Path

from websockets.asyncio.client import connect


SERVER_IP = "192.168.152.207"
SERVER_PORT = 8765
URI = f"wss://{SERVER_IP}:{SERVER_PORT}"

NUM_TRIALS = 20
WAIT_BETWEEN_TRIALS = 0.5

DATASET_DIR = Path("../dataset")

OUTPUT_FILE = "connection_and_transmission_results.txt"


async def main():

    # -----------------------------
    # Select file from dataset
    # -----------------------------
    files = [
        file for file in DATASET_DIR.iterdir()
        if file.is_file()
    ]

    if not files:
        print(f"No files found in {DATASET_DIR}")
        return

    # Select the first file
    file_path = files[20]

    # Read file once
    with open(file_path, "rb") as f:
        data = f.read()

    data_size = len(data)

    print(f"Selected file: {file_path}")
    print(f"File size: {data_size} bytes")
    print("-" * 50)

    # -----------------------------
    # SSL configuration
    # -----------------------------
    ssl_context = ssl.create_default_context(
        cafile="server.crt"
    )

    connection_results = []
    transmission_results = []

    # -----------------------------
    # Output file
    # -----------------------------
    with open(OUTPUT_FILE, "a") as f:

        f.write(f"Server: {SERVER_IP}\n")
        f.write(f"Port: {SERVER_PORT}\n")
        f.write(f"File: {file_path}\n")
        f.write(f"File size: {data_size} bytes\n")
        f.write("-" * 60 + "\n")

        # -----------------------------
        # Trials
        # -----------------------------
        for i in range(NUM_TRIALS):

            # =============================
            # Connection time
            # =============================

            connection_start = time.perf_counter()

            async with connect(
                URI,
                ssl=ssl_context
            ) as websocket:

                connection_end = time.perf_counter()

                connection_delay_ms = (
                    connection_end - connection_start
                ) * 1000

                connection_results.append(
                    connection_delay_ms
                )

                # =============================
                # Data transmission
                # =============================

                transmission_start = time.perf_counter()

                await websocket.send(data)

                # Wait for server ACK
                response = await websocket.recv()

                transmission_end = time.perf_counter()

                transmission_delay_ms = (
                    transmission_end - transmission_start
                ) * 1000

                transmission_results.append(
                    transmission_delay_ms
                )

            # =============================
            # Print results
            # =============================

            print(
                f"Trial {i + 1:03d}: "
                f"Connection = {connection_delay_ms:.3f} ms, "
                f"Transmission = {transmission_delay_ms:.3f} ms"
            )

            # =============================
            # Write results
            # =============================

            f.write(
                f"Trial {i + 1:03d}: "
                f"Connection = {connection_delay_ms:.3f} ms, "
                f"Transmission = {transmission_delay_ms:.3f} ms\n"
            )

            # Wait before next connection
            # await asyncio.sleep(WAIT_BETWEEN_TRIALS)

        # -----------------------------
        # Statistics
        # -----------------------------

        avg_connection = (
            sum(connection_results)
            / len(connection_results)
        )

        min_connection = min(connection_results)
        max_connection = max(connection_results)

        avg_transmission = (
            sum(transmission_results)
            / len(transmission_results)
        )

        min_transmission = min(transmission_results)
        max_transmission = max(transmission_results)

        # -----------------------------
        # Print statistics
        # -----------------------------

        print("\nResults")

        print(
            f"Connection - Average: "
            f"{avg_connection:.3f} ms"
        )

        print(
            f"Connection - Minimum: "
            f"{min_connection:.3f} ms"
        )

        print(
            f"Connection - Maximum: "
            f"{max_connection:.3f} ms"
        )

        print()

        print(
            f"Transmission - Average: "
            f"{avg_transmission:.3f} ms"
        )

        print(
            f"Transmission - Minimum: "
            f"{min_transmission:.3f} ms"
        )

        print(
            f"Transmission - Maximum: "
            f"{max_transmission:.3f} ms"
        )

        # -----------------------------
        # Write statistics
        # -----------------------------

        f.write("\nResults\n")

        f.write(
            f"Connection Average: "
            f"{avg_connection:.3f} ms\n"
        )

        f.write(
            f"Connection Minimum: "
            f"{min_connection:.3f} ms\n"
        )

        f.write(
            f"Connection Maximum: "
            f"{max_connection:.3f} ms\n"
        )

        f.write("\n")

        f.write(
            f"Transmission Average: "
            f"{avg_transmission:.3f} ms\n"
        )

        f.write(
            f"Transmission Minimum: "
            f"{min_transmission:.3f} ms\n"
        )

        f.write(
            f"Transmission Maximum: "
            f"{max_transmission:.3f} ms\n"
        )


if __name__ == "__main__":
    asyncio.run(main())