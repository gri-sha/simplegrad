"""WebSocket management for real-time updates."""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

ws_router = APIRouter()


class WebSocketManager:
    """Manage WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: dict[int, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, run_id: int):
        await websocket.accept()
        if run_id not in self.active_connections:
            self.active_connections[run_id] = []
        self.active_connections[run_id].append(websocket)

    def disconnect(self, websocket: WebSocket, run_id: int):
        if run_id in self.active_connections:
            self.active_connections[run_id].remove(websocket)
            if not self.active_connections[run_id]:
                del self.active_connections[run_id]

    async def broadcast(self, run_id: int, message: dict):
        """Broadcast message to all connections watching a run."""
        if run_id in self.active_connections:
            for connection in self.active_connections[run_id]:
                try:
                    await connection.send_json(message)
                except Exception:
                    pass


ws_manager = WebSocketManager()


@ws_router.websocket("/ws/{run_id}")
async def websocket_endpoint(websocket: WebSocket, run_id: int):
    """WebSocket endpoint for real-time metric updates."""
    await ws_manager.connect(websocket, run_id)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, run_id)
