"""FastAPI server and WebSocket handling."""
import asyncio
import json
import time
from typing import Optional, Dict, Any, List, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
from loguru import logger

from ..config import get_settings
from ..core import CameraCapture, Frame
from ..slam import DroidSLAM, PointCloudMap
from ..cv import SAMSegmenter, CLIPEmbedder
from ..scene import SemanticMap, SemanticSearch, SceneGraph
from ..human import HumanTracker, HumanDigitalTwin
from ..prediction import FuturePredictor, ActionPredictor
from ..physics import PhysicsEngine
from ..graph import Neo4jStore, EventTracker
from ..llm import LLMAgent, SceneContext


# Pydantic models for API
class PoseModel(BaseModel):
    position: Dict[str, float]
    orientation: Dict[str, float]


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    results: List[Dict[str, Any]]
    query: str
    total: int


class QueryRequest(BaseModel):
    question: str
    context: bool = True


class QueryResponse(BaseModel):
    response: str
    confidence: float
    sources: List[str]


class PhysicsRequest(BaseModel):
    object_id: str
    action: str
    parameters: Dict[str, Any]


# Connection manager for WebSocket
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.subscribers: Dict[str, List[WebSocket]] = {
            "camera": [],
            "scene": [],
            "human": [],
            "prediction": []
        }

    async def connect(self, websocket: WebSocket, channel: str = "general"):
        await websocket.accept()
        self.active_connections.append(websocket)
        if channel in self.subscribers:
            self.subscribers[channel].append(websocket)

    def disconnect(self, websocket: WebSocket, channel: str = "general"):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if channel in self.subscribers and websocket in self.subscribers[channel]:
            self.subscribers[channel].remove(websocket)

    async def broadcast(self, message: Dict[str, Any], channel: str = "general"):
        disconnected = []
        for connection in self.subscribers.get(channel, []):
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)

        # Clean up disconnected
        for ws in disconnected:
            self.disconnect(ws, channel)


manager = ConnectionManager()


# Global state
class AXISState:
    def __init__(self):
        self.camera: Optional[CameraCapture] = None
        self.slam: Optional[DroidSLAM] = None
        self.map: Optional[PointCloudMap] = None
        self.semantic_map: Optional[SemanticMap] = None
        self.scene_graph: Optional[SceneGraph] = None
        self.search: Optional[SemanticSearch] = None
        self.segmenter: Optional[SAMSegmenter] = None
        self.feature_extractor: Optional[CLIPEmbedder] = None
        self.human_tracker: Optional[HumanTracker] = None
        self.digital_twin: Optional[HumanDigitalTwin] = None
        self.predictor: Optional[FuturePredictor] = None
        self.action_predictor: Optional[ActionPredictor] = None
        self.physics: Optional[PhysicsEngine] = None
        self.graph_store: Optional[Neo4jStore] = None
        self.event_tracker: Optional[EventTracker] = None
        self.llm_agent: Optional[LLMAgent] = None

    def initialize(self, settings):
        """Initialize all components."""
        logger.info("Initializing AXIS components...")

        # Camera
        self.camera = CameraCapture(
            device_id=0,
            width=settings.camera_width,
            height=settings.camera_height,
            fps=settings.camera_fps
        )

        # SLAM
        self.slam = DroidSLAM(use_depth=False)

        # Maps
        self.map = PointCloudMap()
        self.semantic_map = SemanticMap()
        self.scene_graph = SceneGraph()
        self.search = SemanticSearch(self.semantic_map)

        # CV Models
        self.segmenter = SAMSegmenter(device="cuda" if settings.cuda_enabled else "cpu")
        self.feature_extractor = CLIPEmbedder(device="cuda" if settings.cuda_enabled else "cpu")
        self.search.set_embedding_model(self.feature_extractor)

        # Human
        self.human_tracker = HumanTracker()
        self.digital_twin = HumanDigitalTwin()

        # Prediction
        self.predictor = FuturePredictor()
        self.action_predictor = ActionPredictor()

        # Physics
        self.physics = PhysicsEngine()

        # Graph store
        self.graph_store = Neo4jStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password
        )
        self.event_tracker = EventTracker(self.graph_store)

        # LLM
        self.llm_agent = LLMAgent(
            api_key=settings.openai_api_key,
            model_name=settings.llm_model
        )

        logger.info("AXIS initialization complete")


# Global state instance
axis_state = AXISState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    settings = get_settings()
    axis_state.initialize(settings)

    yield

    # Cleanup
    if axis_state.camera:
        axis_state.camera.stop()
    logger.info("AXIS shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="AXIS API",
    description="Adaptive eXtended Intelligence System - Embodied AI Platform",
    version="0.1.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# WebSocket endpoint
@app.websocket("/ws/{channel}")
async def websocket_endpoint(websocket: WebSocket, channel: str):
    """WebSocket for real-time updates."""
    await manager.connect(websocket, channel)
    logger.info(f"WebSocket connected: {channel}")

    try:
        while True:
            data = await websocket.receive_json()
            
            # Handle commands
            cmd = data.get("command")
            
            if cmd == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})
            
            elif cmd == "subscribe":
                # Already subscribed via path
                pass
            
            elif cmd == "get_state":
                await websocket.send_json(axis_state_to_dict())
            
            elif cmd == "search":
                results = await handle_search(data.get("query", ""))
                await websocket.send_json({"type": "search_results", "results": results})
            
            elif cmd == "query":
                response = await handle_llm_query(data.get("question", ""))
                await websocket.send_json({"type": "llm_response", "response": response})

    except WebSocketDisconnect:
        manager.disconnect(websocket, channel)
        logger.info(f"WebSocket disconnected: {channel}")


def axis_state_to_dict() -> Dict[str, Any]:
    """Get current state as dictionary."""
    return {
        "camera_running": axis_state.camera.is_running if axis_state.camera else False,
        "slam_state": axis_state.slam.state.value if axis_state.slam else "idle",
        "map_points": axis_state.map.point_count if axis_state.map else 0,
        "semantic_objects": len(axis_state.semantic_map.objects) if axis_state.semantic_map else 0,
        "keyframes": axis_state.slam.keyframe_count if axis_state.slam else 0
    }


async def handle_search(query: str) -> List[Dict[str, Any]]:
    """Handle semantic search request."""
    if axis_state.search is None:
        return []
    
    results = axis_state.search.search_by_text(query, top_k=5)
    return [r.to_dict() for r in results]


async def handle_llm_query(question: str) -> Dict[str, Any]:
    """Handle LLM query request."""
    if axis_state.llm_agent is None:
        return {"response": "LLM not available", "confidence": 0}

    context = SceneContext(
        objects=[obj.to_dict() for obj in axis_state.semantic_map.objects.values()],
        persons=[],  # Would include tracked humans
        relationships=[],  # From scene graph
        recent_events=[],  # From event tracker
        spatial_info={}
    )

    result = axis_state.llm_agent.query(question, context)
    return result.to_dict()


# REST API endpoints
@app.get("/")
async def root():
    return {"message": "AXIS API", "version": "0.1.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": time.time()}


@app.get("/state")
async def get_state():
    return axis_state_to_dict()


@app.post("/camera/start")
async def start_camera():
    if axis_state.camera:
        axis_state.camera.start()
        return {"status": "started"}
    return {"status": "error", "message": "Camera not initialized"}


@app.post("/camera/stop")
async def stop_camera():
    if axis_state.camera:
        axis_state.camera.stop()
        return {"status": "stopped"}
    return {"status": "error", "message": "Camera not initialized"}


@app.get("/camera/frame")
async def get_frame():
    if axis_state.camera:
        frame = axis_state.camera.get_frame(timeout=1.0)
        if frame:
            return {
                "frame_id": frame.frame_id,
                "timestamp": frame.timestamp,
                "shape": frame.rgb.shape
            }
    return {"error": "No frame available"}


@app.post("/search")
async def search(request: SearchRequest) -> SearchResponse:
    results = await handle_search(request.query)
    return SearchResponse(
        results=results,
        query=request.query,
        total=len(results)
    )


@app.post("/query")
async def query(request: QueryRequest) -> QueryResponse:
    result = await handle_llm_query(request.question)
    return QueryResponse(
        response=result.get("response", ""),
        confidence=result.get("confidence", 0),
        sources=result.get("context_used", [])
    )


@app.post("/physics/simulate")
async def simulate_physics(request: PhysicsRequest):
    if axis_state.physics is None:
        raise HTTPException(status_code=500, detail="Physics engine not available")

    result = axis_state.physics.simulate_interaction(
        request.object_id,
        request.action,
        request.parameters
    )

    return {
        "success": result.success,
        "states": result.object_states,
        "collisions": result.collision_events
    }


@app.get("/trajectory")
async def get_trajectory():
    if axis_state.slam is None:
        return {"trajectory": []}
    
    trajectory = axis_state.slam.get_trajectory()
    return {"trajectory": trajectory.tolist()}


@app.get("/map/points")
async def get_map_points():
    if axis_state.map is None:
        return {"points": []}
    
    points, colors = axis_state.map.get_point_cloud()
    return {
        "points": points.tolist(),
        "colors": colors.tolist() if len(colors) > 0 else []
    }


@app.get("/semantic/objects")
async def get_semantic_objects():
    if axis_state.semantic_map is None:
        return {"objects": []}
    
    return {
        "objects": [obj.to_dict() for obj in axis_state.semantic_map.objects.values()],
        "stats": axis_state.semantic_map.get_stats()
    }


@app.get("/scene/graph")
async def get_scene_graph():
    if axis_state.scene_graph is None:
        return {"nodes": [], "edges": []}
    
    return axis_state.scene_graph.to_dict()


# Export app for uvicorn
router = app