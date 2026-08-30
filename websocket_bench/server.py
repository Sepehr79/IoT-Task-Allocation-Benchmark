import asyncio
import ssl

from websockets.asyncio.server import serve


HOST = "0.0.0.0"
PORT = 8765


async def handler(websocket):
    print("Client connected")

    try:
        async for message in websocket:
            print(f"Received: {message}")

            await websocket.send(
                f"Server received: {message}"
            )

    except Exception as e:
        print(f"Connection error: {e}")

    finally:
        print("Client disconnected")


async def main():

    ssl_context = ssl.SSLContext(
        ssl.PROTOCOL_TLS_SERVER
    )

    ssl_context.load_cert_chain(
        certfile="server.crt",
        keyfile="server.key"
    )

    async with serve(
        handler,
        HOST,
        PORT,
        ssl=ssl_context
    ):
        print(
            f"WSS server listening on "
            f"wss://{HOST}:{PORT}"
        )

        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())