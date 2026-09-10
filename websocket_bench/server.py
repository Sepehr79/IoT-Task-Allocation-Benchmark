import asyncio

import ssl

from websockets.asyncio.server import serve


HOST = "0.0.0.0"
PORT = 8765


async def handler(websocket):

    print("Client connected")

    try:

        async for message in websocket:

            # -----------------------------
            # Binary message
            # -----------------------------

            if isinstance(message, bytes):

                data_size = len(message)

                print(
                    f"Received binary data: "
                    f"{data_size} bytes"
                )

                # Send ACK
                await websocket.send(
                    f"ACK: {data_size} bytes received"
                )

            # -----------------------------
            # Text message
            # -----------------------------

            else:

                data_size = len(
                    message.encode("utf-8")
                )

                print(
                    f"Received text data: "
                    f"{data_size} bytes"
                )

                await websocket.send(
                    f"ACK: {data_size} bytes received"
                )

    except Exception as e:

        print(
            f"Connection error: {e}"
        )

    finally:

        print("Client disconnected")


async def main():

    # -----------------------------
    # SSL configuration
    # -----------------------------

    ssl_context = ssl.SSLContext(
        ssl.PROTOCOL_TLS_SERVER
    )

    ssl_context.load_cert_chain(
        certfile="server.crt",
        keyfile="server.key"
    )

    # -----------------------------
    # Start WSS server
    # -----------------------------

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