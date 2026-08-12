import asyncio
import ssl
from pathlib import Path

import websockets


HOST = "0.0.0.0"
PORT = 8765


async def handle_user(websocket):
    print("User connected")

    try:
        async for message in websocket:
            print(f"Received task: {message}")

            # Process the task here
            result = f"Task processed: {message}"

            await websocket.send(result)

    except websockets.ConnectionClosed:
        print("User disconnected")


async def main():
    base_dir = Path(__file__).resolve().parent

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    ssl_context.load_cert_chain(
        certfile=base_dir / "server.crt",
        keyfile=base_dir / "server.key"
    )

    async with websockets.serve(
        handle_user,
        HOST,
        PORT,
        ssl=ssl_context
    ):
        print(f"Secure WebSocket server running on wss://localhost:{PORT}")

        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())