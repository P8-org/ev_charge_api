
from asyncio import Lock
from fastapi import WebSocket


class WSManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.lock = Lock()  # Add a lock for concurrency control

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self.lock:
            self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        async with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        print("active ws connections:", len(self.active_connections))
        async with self.lock:
            for connection in self.active_connections:
                try:
                    await connection.send_text(message)
                except:
                    await self.disconnect(connection)