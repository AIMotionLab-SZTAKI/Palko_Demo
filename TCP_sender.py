import trio

async def sender():
    stream: trio.SocketStream = await trio.open_tcp_stream("127.0.0.1", 6001)
    async with stream:
        while True:
            data = input("Drone ID: ")
            await stream.send_all(data.encode("utf-8"))

trio.run(sender)