import asyncio
import ssl
import time

import websockets

import socket

_orig_getaddrinfo = socket.getaddrinfo

def ipv4_only_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    return _orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

socket.getaddrinfo = ipv4_only_getaddrinfo


URI = "wss://localhost:8765"


async def benchmark():

    ssl_context = ssl.create_default_context(
        cafile="server.crt"
    )

    results = []

    for i in range(1):

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
    print(f"{sum(results) / len(results):.3f} ms")


asyncio.run(benchmark())