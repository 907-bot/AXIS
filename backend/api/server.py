from __future__ import annotations

"""Runnable AXIS MVP server with live scene state and browser-assisted detection."""

import math
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pydantic import BaseModel, Field

from config import get_settings
from backend.core.types import Vector3
from backend.scene import SceneGraph, SemanticMap, SemanticSearch
from backend.scene.scene_graph import RelationType


class BoundingBoxModel(BaseModel):
    x: float
    y: float
    width: float
    height: float


class DetectionModel(BaseModel):
    label: str
    score: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBoxModel


class SceneUpdateRequest(BaseModel):
    frame_id: int = 0
    detections: List[DetectionModel] = Field(default_factory=list)
    camera_active: bool = True
    reset_scene: bool = False


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


class ConnectionManager:
    def __init__(self) -> None:
        self.connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.connections:
            self.connections.remove(websocket)

    async def broadcast(self, payload: Dict[str, Any]) -> None:
        stale: List[WebSocket] = []
        for websocket in self.connections:
            try:
                await websocket.send_json(payload)
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(websocket)


class SimpleEmbeddingModel:
    """Small text embedding fallback so semantic search still works without CLIP."""

    def __init__(self, dim: int = 96) -> None:
        self.dim = dim
        self.synonyms = {
            "person": "person",
            "man": "person",
            "woman": "person",
            "bottle": "bottle",
            "cup": "cup",
            "mug": "cup",
            "glass": "cup",
            "cell phone": "phone",
            "phone": "phone",
            "laptop": "laptop",
            "keyboard": "keyboard",
            "mouse": "mouse",
            "book": "book",
            "backpack": "bag",
            "handbag": "bag",
            "bag": "bag",
            "chair": "chair",
            "couch": "sofa",
            "sofa": "sofa",
            "tv": "tv",
            "monitor": "monitor",
            "remote": "remote",
            "banana": "banana",
            "apple": "apple",
            "orange": "orange",
            "carrot": "carrot",
            "broccoli": "broccoli",
            "potted plant": "plant",
            "plant": "plant",
            "clock": "clock",
            "sink": "sink",
            "bed": "bed",
            "table": "table",
            "dining table": "table",
            "desk": "table",
            "bench": "bench",
            "toilet": "toilet",
            "refrigerator": "fridge",
            "fridge": "fridge",
            "microwave": "microwave",
            "oven": "oven",
            "teddy bear": "toy",
            "scissors": "scissors",
        }

    def normalize_label(self, label: str) -> str:
        token = label.strip().lower()
        return self.synonyms.get(token, token)

    def extract_text(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dim, dtype=np.float32)
        cleaned = text.lower().replace("?", " ").replace(",", " ")
        tokens = [self.normalize_label(piece) for piece in cleaned.split() if piece.strip()]
        if not tokens:
            return vector

        for token in tokens:
            slot = abs(hash(token)) % self.dim
            vector[slot] += 1.0
            for char in token:
                vector[(slot + ord(char)) % self.dim] += 0.15

        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


class AxisSceneState:
    def __init__(self) -> None:
        self.embedding_model = SimpleEmbeddingModel()
        self.semantic_map = SemanticMap()
        self.scene_graph = SceneGraph()
        self.search = SemanticSearch(self.semantic_map)
        self.search.set_embedding_model(self.embedding_model)
        self.manager = ConnectionManager()
        self.object_counter = 0
        self.frame_count = 0
        self.last_update = 0.0
        self.camera_enabled = False
        self.trajectory: List[Dict[str, float]] = []
        self.events: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.semantic_map.clear()
        self.scene_graph = SceneGraph()
        self.search = SemanticSearch(self.semantic_map)
        self.search.set_embedding_model(self.embedding_model)
        self.object_counter = 0
        self.frame_count = 0
        self.last_update = 0.0
        self.camera_enabled = False
        self.trajectory.clear()
        self.events.clear()

    def _append_event(self, description: str, event_type: str = "info") -> None:
        self.events.append(
            {
                "type": event_type,
                "description": description,
                "timestamp": time.time(),
            }
        )
        self.events = self.events[-20:]

    def _bbox_to_world(self, bbox: BoundingBoxModel, depth_index: int) -> Tuple[Vector3, Vector3]:
        center_x = bbox.x + (bbox.width / 2.0)
        center_y = bbox.y + (bbox.height / 2.0)
        norm_x = (center_x / 640.0) - 0.5
        norm_y = 0.5 - (center_y / 480.0)
        area_ratio = max(0.001, min(1.0, (bbox.width * bbox.height) / (640.0 * 480.0)))
        depth = max(0.6, 4.2 - (area_ratio * 8.0) - (depth_index * 0.18))
        size = Vector3(
            x=max(0.15, (bbox.width / 640.0) * 1.8),
            y=max(0.15, (bbox.height / 480.0) * 1.8),
            z=max(0.12, (bbox.width / 640.0) * 0.9),
        )
        position = Vector3(x=norm_x * 6.0, y=norm_y * 4.0, z=depth)
        return position, size

    def _find_existing_object(self, label: str, position: Vector3, used_ids: set[str]) -> Optional[str]:
        best_id: Optional[str] = None
        best_distance = 999.0
        for object_id, obj in self.semantic_map.objects.items():
            if object_id in used_ids or obj.class_name != label:
                continue
            distance = math.sqrt(
                (obj.position.x - position.x) ** 2
                + (obj.position.y - position.y) ** 2
                + (obj.position.z - position.z) ** 2
            )
            if distance < 1.35 and distance < best_distance:
                best_id = object_id
                best_distance = distance
        return best_id

    def _update_camera_trajectory(self, frame_id: int) -> None:
        angle = frame_id * 0.08
        camera_position = {
            "x": round(math.sin(angle) * 0.35, 3),
            "y": round(1.55 + math.cos(angle * 0.5) * 0.08, 3),
            "z": round(3.4 + math.cos(angle) * 0.25, 3),
        }
        self.trajectory.append(camera_position)
        self.trajectory = self.trajectory[-120:]

    def _rebuild_scene_graph(self) -> None:
        graph = SceneGraph()
        objects = list(self.semantic_map.objects.values())

        for obj in objects:
            graph.add_node(
                node_id=obj.object_id,
                label=obj.class_name,
                node_type="object",
                position=obj.position,
                confidence=obj.confidence,
                observations=obj.observation_count,
            )

        for index, left in enumerate(objects):
            for right in objects[index + 1 :]:
                dx = right.position.x - left.position.x
                dy = right.position.y - left.position.y
                dz = right.position.z - left.position.z
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                if distance < 1.8:
                    confidence = max(0.35, 1.0 - (distance / 2.2))
                    graph.add_edge(left.object_id, right.object_id, RelationType.NEAR, confidence=confidence)
                    graph.add_edge(right.object_id, left.object_id, RelationType.NEAR, confidence=confidence)
                if dx > 0.45:
                    graph.add_edge(left.object_id, right.object_id, RelationType.LEFT_OF, confidence=0.75)
                    graph.add_edge(right.object_id, left.object_id, RelationType.RIGHT_OF, confidence=0.75)
                elif dx < -0.45:
                    graph.add_edge(left.object_id, right.object_id, RelationType.RIGHT_OF, confidence=0.75)
                    graph.add_edge(right.object_id, left.object_id, RelationType.LEFT_OF, confidence=0.75)
                if dy > 0.35:
                    graph.add_edge(left.object_id, right.object_id, RelationType.BELOW, confidence=0.7)
                    graph.add_edge(right.object_id, left.object_id, RelationType.ABOVE, confidence=0.7)
                elif dy < -0.35:
                    graph.add_edge(left.object_id, right.object_id, RelationType.ABOVE, confidence=0.7)
                    graph.add_edge(right.object_id, left.object_id, RelationType.BELOW, confidence=0.7)

        self.scene_graph = graph

    def _remove_stale_objects(self, max_age_seconds: float = 8.0) -> None:
        now = time.time()
        stale_ids = [
            object_id
            for object_id, obj in self.semantic_map.objects.items()
            if (now - obj.last_observed) > max_age_seconds
        ]
        for object_id in stale_ids:
            self.semantic_map.remove_object(object_id)

    def ingest(self, payload: SceneUpdateRequest) -> Dict[str, Any]:
        if payload.reset_scene:
            self.reset()
            self._append_event("Scene reset", "system")

        self.frame_count = max(self.frame_count + 1, payload.frame_id)
        self.last_update = time.time()
        self.camera_enabled = payload.camera_active
        self._update_camera_trajectory(self.frame_count)

        used_ids: set[str] = set()
        detections = sorted(payload.detections, key=lambda item: item.score, reverse=True)[:10]
        new_labels: List[str] = []

        for depth_index, detection in enumerate(detections):
            label = self.embedding_model.normalize_label(detection.label)
            position, size = self._bbox_to_world(detection.bbox, depth_index)
            existing_id = self._find_existing_object(label, position, used_ids)
            embedding = self.embedding_model.extract_text(label)

            if existing_id is None:
                self.object_counter += 1
                object_id = f"{label}-{self.object_counter}"
                self.semantic_map.add_object(
                    object_id=object_id,
                    class_name=label,
                    position=position,
                    size=size,
                    clip_embedding=embedding,
                    confidence=detection.score,
                )
                used_ids.add(object_id)
                new_labels.append(label)
            else:
                self.semantic_map.update_object(
                    existing_id,
                    position=position,
                    confidence=detection.score,
                    clip_embedding=embedding,
                    timestamp=self.last_update,
                )
                used_ids.add(existing_id)

        self._remove_stale_objects()
        self._rebuild_scene_graph()

        if new_labels:
            pretty = ", ".join(new_labels[:4])
            self._append_event(f"Detected: {pretty}", "detection")

        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        recent = (time.time() - self.last_update) < 3.0
        objects = [obj.to_dict() for obj in self.semantic_map.objects.values()]
        return {
            "type": "state",
            "camera_running": self.camera_enabled and recent,
            "slam_state": "tracking" if recent and objects else ("scanning" if recent else "idle"),
            "map_points": len(self.semantic_map.objects) * 32,
            "semantic_objects": len(self.semantic_map.objects),
            "keyframes": len(self.trajectory),
            "trajectory": self.trajectory[-60:],
            "objects": objects,
            "scene_graph": self.scene_graph.to_dict(),
            "recent_events": list(reversed(self.events[-8:])),
            "stats": self.semantic_map.get_stats(),
        }

    def semantic_objects(self) -> Dict[str, Any]:
        return {
            "objects": [obj.to_dict() for obj in self.semantic_map.objects.values()],
            "stats": self.semantic_map.get_stats(),
        }

    def map_points(self) -> Dict[str, Any]:
        points: List[List[float]] = []
        colors: List[List[float]] = []
        for obj in self.semantic_map.objects.values():
            seed = abs(hash(obj.object_id)) % (2**32)
            rng = np.random.default_rng(seed)
            for _ in range(22):
                points.append(
                    [
                        float(obj.position.x + rng.normal(scale=max(0.04, obj.size.x * 0.2))),
                        float(obj.position.y + rng.normal(scale=max(0.04, obj.size.y * 0.2))),
                        float(obj.position.z + rng.normal(scale=max(0.04, obj.size.z * 0.2))),
                    ]
                )
                colors.append(
                    [
                        min(1.0, 0.25 + obj.confidence * 0.7),
                        0.75,
                        1.0 - min(0.8, obj.confidence * 0.5),
                    ]
                )
        return {"points": points, "colors": colors}

    def search_objects(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not self.semantic_map.objects:
            return []

        lowered = query.lower().strip()
        lexical_matches = []
        for obj in self.semantic_map.objects.values():
            obj_name = obj.class_name.lower()
            if lowered in obj_name or obj_name in lowered:
                lexical_matches.append(
                    {
                        "object_id": obj.object_id,
                        "class_name": obj.class_name,
                        "position": {"x": obj.position.x, "y": obj.position.y, "z": obj.position.z},
                        "similarity": 0.99,
                        "confidence": obj.confidence,
                    }
                )

        if lexical_matches:
            return lexical_matches[:top_k]

        return [result.to_dict() for result in self.search.search_by_text(query, top_k=top_k)]

    def _describe_position(self, position: Dict[str, float]) -> str:
        horizontal = "center"
        vertical = "middle"
        depth = "mid-depth"

        if position["x"] < -1.4:
            horizontal = "far left"
        elif position["x"] < -0.45:
            horizontal = "left"
        elif position["x"] > 1.4:
            horizontal = "far right"
        elif position["x"] > 0.45:
            horizontal = "right"

        if position["y"] > 0.8:
            vertical = "upper area"
        elif position["y"] < -0.8:
            vertical = "lower area"

        if position["z"] < 1.2:
            depth = "very close to the camera"
        elif position["z"] < 2.2:
            depth = "close to the camera"
        elif position["z"] > 3.4:
            depth = "deeper in the scene"

        return f"{horizontal}, {vertical}, {depth}"

    def answer_question(self, question: str) -> Dict[str, Any]:
        lowered = question.lower().strip()
        objects = list(self.semantic_map.objects.values())
        class_counts = self.semantic_map.get_stats().get("class_counts", {})

        if not objects:
            return {
                "response": "I am not seeing any recognized objects yet. Start the camera and let the detector scan the scene for a couple of seconds.",
                "confidence": 0.2,
                "sources": [],
            }

        if "where" in lowered:
            matches = self.search_objects(question, top_k=1)
            if matches:
                match = matches[0]
                description = self._describe_position(match["position"])
                return {
                    "response": f"I can see a {match['class_name']} around the {description}.",
                    "confidence": round(max(0.45, match["confidence"]), 2),
                    "sources": [match["object_id"]],
                }

        if "how many" in lowered:
            for class_name, count in class_counts.items():
                if class_name in lowered:
                    noun = class_name if count == 1 else f"{class_name}s"
                    return {
                        "response": f"I currently detect {count} {noun} in view.",
                        "confidence": 0.86,
                        "sources": [class_name],
                    }

        if "what do you see" in lowered or "describe" in lowered or "scene" in lowered:
            summary = ", ".join(f"{count} {label}" for label, count in list(class_counts.items())[:6])
            return {
                "response": f"I currently recognize {summary}. The scene graph is updating live as detections arrive.",
                "confidence": 0.84,
                "sources": list(class_counts.keys())[:6],
            }

        if "what changed" in lowered or "recent" in lowered:
            if self.events:
                recent_descriptions = "; ".join(event["description"] for event in self.events[-3:])
                return {
                    "response": f"Recent updates: {recent_descriptions}.",
                    "confidence": 0.74,
                    "sources": [event["type"] for event in self.events[-3:]],
                }

        if "predict" in lowered or "next" in lowered or "will happen" in lowered:
            moving = [obj.class_name for obj in objects if not obj.is_static]
            if moving:
                prediction = f"I expect {moving[0]} to keep moving if the camera continues tracking it."
            else:
                prediction = "If the scene stays stable, the current object layout should remain similar over the next few seconds."
            return {
                "response": prediction,
                "confidence": 0.55,
                "sources": moving[:3],
            }

        top_objects = ", ".join(obj.class_name for obj in objects[:5])
        return {
            "response": f"I am tracking these objects right now: {top_objects}. Ask where something is, how many items I see, or what changed.",
            "confidence": 0.68,
            "sources": [obj.object_id for obj in objects[:5]],
        }


axis_state = AxisSceneState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AXIS MVP server starting")
    yield
    logger.info("AXIS MVP server shutting down")


app = FastAPI(
    title="AXIS API",
    description="Adaptive eXtended Intelligence System MVP",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

settings = get_settings()
frontend_dir = Path(settings.base_dir) / "frontend"
if frontend_dir.exists():
    app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")


@app.get("/")
async def root():
    index_path = frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"message": "AXIS API", "version": "0.2.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "objects": len(axis_state.semantic_map.objects),
    }


@app.get("/state")
async def get_state():
    return axis_state.snapshot()


@app.post("/scene/update")
async def update_scene(payload: SceneUpdateRequest):
    snapshot = axis_state.ingest(payload)
    await axis_state.manager.broadcast(snapshot)
    return snapshot


@app.post("/scene/reset")
async def reset_scene():
    axis_state.reset()
    axis_state._append_event("Scene reset", "system")
    snapshot = axis_state.snapshot()
    await axis_state.manager.broadcast(snapshot)
    return snapshot


@app.post("/camera/start")
async def start_camera():
    axis_state.camera_enabled = True
    axis_state._append_event("Camera stream enabled from browser", "camera")
    snapshot = axis_state.snapshot()
    await axis_state.manager.broadcast(snapshot)
    return {"status": "started"}


@app.post("/camera/stop")
async def stop_camera():
    axis_state.camera_enabled = False
    axis_state._append_event("Camera stream stopped", "camera")
    snapshot = axis_state.snapshot()
    await axis_state.manager.broadcast(snapshot)
    return {"status": "stopped"}


@app.post("/search")
async def search(request: SearchRequest) -> SearchResponse:
    results = axis_state.search_objects(request.query, top_k=request.top_k)
    return SearchResponse(results=results, query=request.query, total=len(results))


@app.post("/query")
async def query(request: QueryRequest) -> QueryResponse:
    result = axis_state.answer_question(request.question)
    return QueryResponse(**result)


@app.get("/trajectory")
async def get_trajectory():
    return {"trajectory": axis_state.trajectory}


@app.get("/map/points")
async def get_map_points():
    return axis_state.map_points()


@app.get("/semantic/objects")
async def get_semantic_objects():
    return axis_state.semantic_objects()


@app.get("/scene/graph")
async def get_scene_graph():
    return axis_state.scene_graph.to_dict()


@app.websocket("/ws/scene")
async def websocket_scene(websocket: WebSocket):
    await axis_state.manager.connect(websocket)
    await websocket.send_json(axis_state.snapshot())
    try:
        while True:
            payload = await websocket.receive_json()
            command = payload.get("command")
            if command == "ping":
                await websocket.send_json({"type": "pong", "timestamp": time.time()})
            elif command == "get_state":
                await websocket.send_json(axis_state.snapshot())
            elif command == "search":
                await websocket.send_json(
                    {
                        "type": "search_results",
                        "results": axis_state.search_objects(payload.get("query", ""), top_k=5),
                    }
                )
            elif command == "query":
                await websocket.send_json(
                    {
                        "type": "llm_response",
                        "response": axis_state.answer_question(payload.get("question", ""))["response"],
                    }
                )
    except WebSocketDisconnect:
        axis_state.manager.disconnect(websocket)


router = app
