"""
FastAPI server for the visualization dashboard.
Provides REST API endpoints for accessing training runs and metrics.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .api import router, ws_router


app = FastAPI(
    title="simpleboard",
    description="training visualization dashboard for simplegrad",
    version="0.1.0",
)

# CORS for development - must be before routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)
app.include_router(ws_router)


# Static File Serving

sb_dist = Path(__file__).parent / "app" / "dist"
print("simpleboard dist:", sb_dist)

if sb_dist.exists():
    app.mount("/assets", StaticFiles(directory=sb_dist / "assets"), name="assets")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(sb_dist / "index.html")

    @app.get("/{path:path}")
    async def serve_frontend_routes(path: str):
        file_path = sb_dist / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(sb_dist / "index.html")

else:
    print("Warning: simpleboard frontend not found. Please build the web app.")
    print("Obtained path:", sb_dist)
