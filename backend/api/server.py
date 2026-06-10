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
from backend.scene import SceneGraph, SemanticMap, SemanticSearch, GaussianMap, SceneIntelligence
from backend.scene.scene_graph import RelationType
from backend.human.analytics import summarize_analytics
from backend.predict import PredictionEngine
from backend.physics import PhysicsEngine, sim_force_vectors
from backend.graph.neo4j_store import Neo4jStore, EventTracker, GraphAnalytics
from backend.llm.reasoning import LLMAgent, SceneContext, ReasoningType
from backend.agent import ReasoningAgent
from backend.events import EventProducer, KafkaConfig

# ---------------------------------------------------------------------------
# Named constants
# ---------------------------------------------------------------------------
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
WORLD_SCALE_X = 6.0
WORLD_SCALE_Y = 4.0
MIN_DEPTH = 0.6
MAX_DEPTH_OFFSET = 4.2
DEPTH_AREA_FACTOR = 8.0
DEPTH_INDEX_STEP = 0.18
MIN_OBJECT_SIZE = 0.15
SIZE_WIDTH_FACTOR = 1.8
SIZE_HEIGHT_FACTOR = 1.8
SIZE_DEPTH_FACTOR = 0.9
MATCH_DISTANCE_THRESHOLD = 1.35
CAMERA_ANGLE_STEP = 0.08
CAMERA_Y_BASE = 1.55
CAMERA_Y_AMPLITUDE = 0.08
CAMERA_Z_BASE = 3.4
CAMERA_Z_AMPLITUDE = 0.25
TRAJECTORY_MAX_LEN = 120
STALE_OBJECT_AGE = 8.0
STALE_GAUSSIAN_AGE = 6.0
MAX_DETECTIONS_PER_FRAME = 10
GAUSSIAN_SCALE_FACTOR = 0.6
GAUSSIAN_OPACITY_BASE = 0.4
SCENE_GRAPH_NEAR_RATIO = 1.8
SCENE_GRAPH_NEAR_MAX_DIST = 2.2
SCENE_GRAPH_LEFT_RIGHT_THRESHOLD = 0.45
SCENE_GRAPH_ABOVE_BELOW_THRESHOLD = 0.35
SCENE_GRAPH_DIRECTION_CONFIDENCE = 0.75
SCENE_GRAPH_VERTICAL_CONFIDENCE = 0.7
RECENT_EVENTS_MAX = 20
SNAPSHOT_RECENT_EVENTS = 8
TRAJECTORY_SNAPSHOT_LEN = 60
HUMAN_POSE_HISTORY_MAX = 120
HUMAN_POSE_ANALYSIS_WINDOW = 30
RANDOM_POINTS_PER_OBJECT = 22
SEARCH_TOP_K_DEFAULT = 5
# ---------------------------------------------------------------------------


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


class KeypointModel(BaseModel):
    x: float
    y: float
    z: float = 0.0
    confidence: float = 1.0


class HumanPoseModel(BaseModel):
    person_id: str
    keypoints: Dict[str, KeypointModel]


class HumanPoseUpdate(BaseModel):
    poses: List[HumanPoseModel]


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


class CLIPEmbeddingModel:
    """Text embedding using sentence-transformers (CLIP-based) with hash fallback."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim
        self._model = None
        self._model_name = "all-MiniLM-L6-v2"
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

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded CLIP model: {self._model_name} (dim={self.dim})")
        except ImportError:
            logger.warning("sentence-transformers not installed, using hash fallback")

    def _stable_hash(self, text: str) -> int:
        import hashlib
        return int(hashlib.md5(text.encode()).hexdigest(), 16)

    def extract_text(self, text: str) -> np.ndarray:
        self._ensure_model()
        if self._model is not None:
            emb = self._model.encode(text, normalize_embeddings=True)
            return np.array(emb, dtype=np.float32)

        vector = np.zeros(self.dim, dtype=np.float32)
        cleaned = text.lower().replace("?", " ").replace(",", " ")
        tokens = [self.normalize_label(piece) for piece in cleaned.split() if piece.strip()]
        if not tokens:
            return vector

        for token in tokens:
            slot = self._stable_hash(token) % self.dim
            vector[slot] += 1.0
            for char in token:
                vector[(slot + ord(char)) % self.dim] += 0.15

        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm


class AxisSceneState:
    def __init__(self) -> None:
        self.embedding_model = CLIPEmbeddingModel()
        self.semantic_map = SemanticMap()
        self.scene_graph = SceneGraph()
        self.search = SemanticSearch(self.semantic_map)
        self.search.set_embedding_model(self.embedding_model)
        self.gaussian_map = GaussianMap()
        self.manager = ConnectionManager()
        self.object_counter = 0
        self.gaussian_counter = 0
        self.frame_count = 0
        self.last_update = 0.0
        self.camera_enabled = False
        self.trajectory: List[Dict[str, float]] = []
        self.events: List[Dict[str, Any]] = []
        self.human_poses: List[Dict[str, Any]] = []
        self.human_pose_history: List[List[Dict[str, Any]]] = []
        self.human_analytics: Dict[str, Any] = {}
        self.predictor = PredictionEngine()
        self.physics = PhysicsEngine()
        self.intelligence = SceneIntelligence()
        self.agent = ReasoningAgent(self.snapshot)

        # Neo4j graph database
        self.neo4j_store: Optional[Neo4jStore] = None
        self.event_tracker: Optional[EventTracker] = None
        self.graph_analytics: Optional[GraphAnalytics] = None
        self._init_neo4j()

        # LangGraph LLM agent
        self.langgraph_agent: Optional[LLMAgent] = None
        self._init_langgraph()

        # Kafka event producer
        self.event_producer = EventProducer()

    def _init_neo4j(self) -> None:
        try:
            self.neo4j_store = Neo4jStore()
            self.event_tracker = EventTracker(self.neo4j_store)
            self.graph_analytics = GraphAnalytics(self.neo4j_store)
            logger.info("Neo4j store initialized")
        except Exception as e:
            logger.warning(f"Neo4j init failed (non-fatal): {e}")

    def _init_langgraph(self) -> None:
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            try:
                self.langgraph_agent = LLMAgent(api_key=api_key)
                logger.info("LangGraph agent initialized")
            except Exception as e:
                logger.warning(f"LangGraph agent init failed (non-fatal): {e}")

    def _sync_scene_graph_to_neo4j(self) -> None:
        if self.neo4j_store is None:
            return
        for node in self.scene_graph.nodes.values():
            self.neo4j_store.create_node(
                node_id=node.node_id,
                labels=["SceneObject", node.node_type.capitalize()],
                properties={
                    "label": node.label,
                    "x": node.position.x if node.position else 0.0,
                    "y": node.position.y if node.position else 0.0,
                    "z": node.position.z if node.position else 0.0,
                    "updated_at": node.updated_at,
                },
            )
        for edge in self.scene_graph.edges.values():
            self.neo4j_store.create_relationship(
                source_id=edge.source_id,
                target_id=edge.target_id,
                rel_type=edge.relation_type.value.upper(),
                properties={"confidence": edge.confidence, "last_observed": edge.last_observed},
            )

    def _build_scene_context(self) -> SceneContext:
        objects = [obj.to_dict() for obj in self.semantic_map.objects.values()]
        persons = self.human_poses
        relationships = []
        for edge in self.scene_graph.edges.values():
            src = self.scene_graph.get_node(edge.source_id)
            tgt = self.scene_graph.get_node(edge.target_id)
            relationships.append({
                "subject": src.label if src else edge.source_id,
                "relation": edge.relation_type.value,
                "object": tgt.label if tgt else edge.target_id,
            })
        return SceneContext(
            objects=objects,
            persons=persons,
            relationships=relationships,
            recent_events=[{"type": e["type"], "description": e["description"]} for e in self.events[-10:]],
            spatial_info={"object_count": len(objects)},
        )

    def reset(self) -> None:
        self.semantic_map.clear()
        self.scene_graph = SceneGraph()
        self.search = SemanticSearch(self.semantic_map)
        self.search.set_embedding_model(self.embedding_model)
        self.gaussian_map.clear()
        self.object_counter = 0
        self.gaussian_counter = 0
        self.frame_count = 0
        self.last_update = 0.0
        self.camera_enabled = False
        self.trajectory.clear()
        self.events.clear()
        self.human_poses.clear()
        self.human_pose_history.clear()
        self.human_analytics.clear()
        self.predictor.reset()
        self.physics.reset()
        self.intelligence.reset()

    def _append_event(self, description: str, event_type: str = "info") -> None:
        timestamp = time.time()
        self.events.append(
            {
                "type": event_type,
                "description": description,
                "timestamp": timestamp,
            }
        )
        self.events = self.events[-RECENT_EVENTS_MAX:]

        # Emit to Kafka
        self.event_producer.emit(event_type, {
            "description": description,
            "timestamp": timestamp,
        })

    def _bbox_to_world(self, bbox: BoundingBoxModel, depth_index: int, frame_width: int = DEFAULT_FRAME_WIDTH, frame_height: int = DEFAULT_FRAME_HEIGHT) -> Tuple[Vector3, Vector3]:
        if frame_width <= 0:
            frame_width = DEFAULT_FRAME_WIDTH
        if frame_height <= 0:
            frame_height = DEFAULT_FRAME_HEIGHT
        center_x = bbox.x + (bbox.width / 2.0)
        center_y = bbox.y + (bbox.height / 2.0)
        norm_x = (center_x / frame_width) - 0.5
        norm_y = 0.5 - (center_y / frame_height)
        area_ratio = max(0.001, min(1.0, (bbox.width * bbox.height) / (frame_width * frame_height)))
        depth = max(MIN_DEPTH, MAX_DEPTH_OFFSET - (area_ratio * DEPTH_AREA_FACTOR) - (depth_index * DEPTH_INDEX_STEP))
        size = Vector3(
            x=max(MIN_OBJECT_SIZE, (bbox.width / frame_width) * SIZE_WIDTH_FACTOR),
            y=max(MIN_OBJECT_SIZE, (bbox.height / frame_height) * SIZE_HEIGHT_FACTOR),
            z=max(0.12, (bbox.width / frame_width) * SIZE_DEPTH_FACTOR),
        )
        position = Vector3(x=norm_x * WORLD_SCALE_X, y=norm_y * WORLD_SCALE_Y, z=depth)
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
            if distance < MATCH_DISTANCE_THRESHOLD and distance < best_distance:
                best_id = object_id
                best_distance = distance
        return best_id

    def _update_camera_trajectory(self, frame_id: int) -> None:
        angle = frame_id * CAMERA_ANGLE_STEP
        camera_position = {
            "x": round(math.sin(angle) * 0.35, 3),
            "y": round(CAMERA_Y_BASE + math.cos(angle * 0.5) * CAMERA_Y_AMPLITUDE, 3),
            "z": round(CAMERA_Z_BASE + math.cos(angle) * CAMERA_Z_AMPLITUDE, 3),
        }
        self.trajectory.append(camera_position)
        self.trajectory = self.trajectory[-TRAJECTORY_MAX_LEN:]

    def _rebuild_scene_graph(self) -> None:
        existing_nodes = set(self.scene_graph.nodes.keys())
        objects = list(self.semantic_map.objects.values())
        current_ids = {o.object_id for o in objects}

        for obj in objects:
            if obj.object_id not in existing_nodes:
                self.scene_graph.add_node(
                    node_id=obj.object_id,
                    label=obj.class_name,
                    node_type="object",
                    position=obj.position,
                    confidence=obj.confidence,
                    observations=obj.observation_count,
                )
            else:
                node = self.scene_graph.get_node(obj.object_id)
                if node:
                    node.position = obj.position
                    node.updated_at = time.time()

        for node_id in existing_nodes:
            if node_id not in current_ids:
                self.scene_graph.remove_node(node_id)

        for index, left in enumerate(objects):
            for right in objects[index + 1 :]:
                dx = right.position.x - left.position.x
                dy = right.position.y - left.position.y
                dz = right.position.z - left.position.z
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)

                if distance < SCENE_GRAPH_NEAR_RATIO:
                    confidence = max(0.35, 1.0 - (distance / SCENE_GRAPH_NEAR_MAX_DIST))
                    self.scene_graph.add_edge(left.object_id, right.object_id, RelationType.NEAR, confidence=confidence)
                    self.scene_graph.add_edge(right.object_id, left.object_id, RelationType.NEAR, confidence=confidence)
                if dx > SCENE_GRAPH_LEFT_RIGHT_THRESHOLD:
                    self.scene_graph.add_edge(left.object_id, right.object_id, RelationType.LEFT_OF, confidence=SCENE_GRAPH_DIRECTION_CONFIDENCE)
                    self.scene_graph.add_edge(right.object_id, left.object_id, RelationType.RIGHT_OF, confidence=SCENE_GRAPH_DIRECTION_CONFIDENCE)
                elif dx < -SCENE_GRAPH_LEFT_RIGHT_THRESHOLD:
                    self.scene_graph.add_edge(left.object_id, right.object_id, RelationType.RIGHT_OF, confidence=SCENE_GRAPH_DIRECTION_CONFIDENCE)
                    self.scene_graph.add_edge(right.object_id, left.object_id, RelationType.LEFT_OF, confidence=SCENE_GRAPH_DIRECTION_CONFIDENCE)
                if dy > SCENE_GRAPH_ABOVE_BELOW_THRESHOLD:
                    self.scene_graph.add_edge(left.object_id, right.object_id, RelationType.BELOW, confidence=SCENE_GRAPH_VERTICAL_CONFIDENCE)
                    self.scene_graph.add_edge(right.object_id, left.object_id, RelationType.ABOVE, confidence=SCENE_GRAPH_VERTICAL_CONFIDENCE)
                elif dy < -SCENE_GRAPH_ABOVE_BELOW_THRESHOLD:
                    self.scene_graph.add_edge(left.object_id, right.object_id, RelationType.ABOVE, confidence=SCENE_GRAPH_VERTICAL_CONFIDENCE)
                    self.scene_graph.add_edge(right.object_id, left.object_id, RelationType.BELOW, confidence=SCENE_GRAPH_VERTICAL_CONFIDENCE)

    def _remove_stale_objects(self, max_age_seconds: float = STALE_OBJECT_AGE) -> None:
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
        detections = sorted(payload.detections, key=lambda item: item.score, reverse=True)[:MAX_DETECTIONS_PER_FRAME]
        if len(payload.detections) > MAX_DETECTIONS_PER_FRAME:
            logger.debug(f"Ingest limited to {MAX_DETECTIONS_PER_FRAME}/{len(payload.detections)} detections")
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

        # Create/update Gaussians from detections
        for depth_index, detection in enumerate(detections):
            label = self.embedding_model.normalize_label(detection.label)
            position, size = self._bbox_to_world(detection.bbox, depth_index)
            hue = (self.embedding_model._stable_hash(label) % 360) / 360.0
            color = (
                min(1.0, 0.3 + 0.7 * (0.5 + 0.5 * math.cos(hue * 2 * math.pi))),
                min(1.0, 0.3 + 0.7 * (0.5 + 0.5 * math.cos((hue + 0.33) * 2 * math.pi))),
                min(1.0, 0.3 + 0.7 * (0.5 + 0.5 * math.cos((hue + 0.67) * 2 * math.pi))),
            )
            self.gaussian_counter += 1
            gaussian_id = f"g_{label}_{self.gaussian_counter}"
            embedding = self.embedding_model.extract_text(label)

            gaussian_scale = Vector3(
                x=max(MIN_OBJECT_SIZE, size.x * GAUSSIAN_SCALE_FACTOR),
                y=max(MIN_OBJECT_SIZE, size.y * GAUSSIAN_SCALE_FACTOR),
                z=max(MIN_OBJECT_SIZE, size.z * GAUSSIAN_SCALE_FACTOR),
            )

            self.gaussian_map.add_gaussian(
                gaussian_id=gaussian_id,
                class_name=label,
                position=position,
                color=color,
                opacity=max(GAUSSIAN_OPACITY_BASE, detection.score * 0.9),
                scale=gaussian_scale,
                clip_embedding=embedding,
                confidence=detection.score,
            )

        self.gaussian_map.remove_stale(max_age_seconds=STALE_GAUSSIAN_AGE)
        self._remove_stale_objects()
        self._rebuild_scene_graph()

        # Scene intelligence analysis
        objects_list = [obj.to_dict() for obj in self.semantic_map.objects.values()]
        scene_intel = self.intelligence.analyze_frame(objects_list, self.human_poses, self.scene_graph)
        for ev in scene_intel.get("events", []):
            if ev.get("description"):
                self._append_event(ev["description"], "intel")

        # Feed observations into prediction engine
        self.predictor.feed_objects([
            {"object_id": oid, "position": obj.position.to_dict(), "class_name": obj.class_name}
            for oid, obj in self.semantic_map.objects.items()
        ])

        if new_labels:
            pretty = ", ".join(new_labels[:4])
            self._append_event(f"Detected: {pretty}", "detection")

        # Sync scene graph to Neo4j
        self._sync_scene_graph_to_neo4j()

        # Record events in Neo4j event tracker
        if self.event_tracker:
            for obj_id in used_ids:
                obj = self.semantic_map.get_object(obj_id)
                if obj:
                    self.event_tracker.record_event(
                        event_type="detection_update",
                        subject_id=obj_id,
                        properties={
                            "class_name": obj.class_name,
                            "confidence": obj.confidence,
                            "timestamp": self.last_update,
                        },
                    )

        return self.snapshot()

    def snapshot(self) -> Dict[str, Any]:
        recent = (time.time() - self.last_update) < 3.0
        objects = [obj.to_dict() for obj in self.semantic_map.objects.values()]
        return {
            "type": "state",
            "camera_running": self.camera_enabled and recent,
            "slam_state": "tracking" if recent and objects else ("scanning" if recent else "idle"),
            "map_points": len(self.semantic_map.objects) * RANDOM_POINTS_PER_OBJECT,
            "semantic_objects": len(self.semantic_map.objects),
            "keyframes": len(self.trajectory),
            "trajectory": self.trajectory[-TRAJECTORY_SNAPSHOT_LEN:],
            "objects": objects,
            "scene_graph": self.scene_graph.to_dict(),
            "gaussians": self.gaussian_map.to_dict_list(),
            "gaussian_count": len(self.gaussian_map.gaussians),
            "recent_events": list(reversed(self.events[-SNAPSHOT_RECENT_EVENTS:])),
            "stats": self.semantic_map.get_stats(),
            "human_poses": self.human_poses[-4:],
            "human_analytics": self.human_analytics,
            "prediction": {
                "action": self.predictor.predict_human_action(),
                "timeline": self.predictor.get_timeline(),
                "stats": self.predictor.get_stats(),
            },
            "physics": self.physics.get_state(),
            "intelligence": {
                "events": [e.to_dict() for e in self.intelligence.events[-15:]],
                "interactions": [i.to_dict() for i in self.intelligence.interactions.values()],
                "event_count": len(self.intelligence.events),
                "interaction_count": len(self.intelligence.interactions),
                "person_tree": self.intelligence.build_person_tree(self.scene_graph),
            },
            "agent": {
                "has_llm": self.agent.has_llm,
                "history_length": len(self.agent.messages),
                "langgraph_available": self.langgraph_agent is not None,
            },
            "neo4j": {
                "available": self.neo4j_store is not None,
                "graph_stats": self.graph_analytics.get_graph_stats() if self.graph_analytics else {},
            },
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
            seed = abs(self.embedding_model._stable_hash(obj.object_id)) % (2**32)
            rng = np.random.default_rng(seed)
            for _ in range(RANDOM_POINTS_PER_OBJECT):
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


_axis_state: Optional[AxisSceneState] = None

def get_state() -> AxisSceneState:
    """Get singleton AxisSceneState instance.

    Using a function instead of a module-level global allows future
    multi-worker deployments to inject a shared state backend (e.g. Redis).
    """
    global _axis_state
    if _axis_state is None:
        _axis_state = AxisSceneState()
    return _axis_state


axis_state = get_state()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("AXIS MVP server starting")
    logger.info(f"Base directory: {settings.base_dir}")
    frontend_dir = Path(settings.base_dir) / "frontend"
    logger.info(f"Frontend directory: {frontend_dir}")
    logger.info(f"Frontend exists: {frontend_dir.exists()}")
    if frontend_dir.exists():
        logger.info(f"Frontend files: {list(frontend_dir.iterdir())}")
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
    allow_credentials=False,
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


@app.post("/human/pose")
async def update_human_pose(payload: HumanPoseUpdate):
    """Receive human pose data from frontend and compute analytics."""
    if payload.poses:
        pose_dicts = []
        for p in payload.poses:
            kp_dict = {
                name: {"x": kp.x, "y": kp.y, "z": kp.z, "confidence": kp.confidence}
                for name, kp in p.keypoints.items()
            }
            pose_dicts.append({"person_id": p.person_id, "keypoints": kp_dict})

        axis_state.human_poses = pose_dicts
        axis_state.human_pose_history.append(pose_dicts)
        if len(axis_state.human_pose_history) > HUMAN_POSE_HISTORY_MAX:
            axis_state.human_pose_history.pop(0)

        if pose_dicts:
            latest = pose_dicts[0]
            flat_history = [
                h[0]["keypoints"] for h in axis_state.human_pose_history[-HUMAN_POSE_ANALYSIS_WINDOW:]
                if h and "keypoints" in h[0]
            ]
            axis_state.human_analytics = summarize_analytics(
                latest["keypoints"], flat_history
            )
            axis_state.predictor.feed_human_pose(latest["keypoints"])
            axis_state._append_event(
                f"Human tracked: {len(pose_dicts)} person(s)", "human"
            )

        # Re-run intelligence with updated poses (only if scene didn't just run it)
        objects_list = [obj.to_dict() for obj in axis_state.semantic_map.objects.values()]
        axis_state.intelligence.analyze_frame(objects_list, axis_state.human_poses, axis_state.scene_graph)

    snapshot = axis_state.snapshot()
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


@app.get("/scene/gaussians")
async def get_gaussians():
    return {
        "gaussians": axis_state.gaussian_map.to_dict_list(),
        "stats": axis_state.gaussian_map.get_stats(),
    }


@app.get("/scene/neural/render")
async def get_neural_scene():
    """Get neural scene representation data for frontend rendering."""
    gaussians = axis_state.gaussian_map.to_dict_list()
    return {
        "gaussians": gaussians,
        "camera": {
            "position": axis_state.trajectory[-1] if axis_state.trajectory else {"x": 0, "y": 1.6, "z": 0},
        },
        "stats": axis_state.gaussian_map.get_stats(),
    }


class PredictFeedRequest(BaseModel):
    detections: List[DetectionModel] = Field(default_factory=list)


@app.post("/predict/feed")
async def predict_feed(payload: PredictFeedRequest):
    """Feed current detections into prediction engine."""
    objects = []
    for d in payload.detections:
        pos, size = axis_state._bbox_to_world(d.bbox, 0)
        objects.append({"object_id": d.label, "position": pos.to_dict(), "class_name": d.label})
    if objects:
        axis_state.predictor.feed_objects(objects)
    return {"status": "fed", "tracked": len(axis_state.predictor.object_buffers)}


@app.get("/predict/timeline")
async def predict_timeline():
    """Get prediction timeline (current, +1s, +3s, +5s)."""
    return axis_state.predictor.get_timeline()


@app.get("/predict/next")
async def predict_next(seconds: float = 3.0):
    """Get prediction for N seconds ahead."""
    return {
        "seconds_ahead": seconds,
        "objects": axis_state.predictor.predict_objects(seconds),
        "human_action": axis_state.predictor.predict_human_action(),
        "human_pose": axis_state.predictor.predict_human_pose(seconds),
    }


class PhysicsInitRequest(BaseModel):
    object_ids: List[str] = Field(default_factory=list)


class PhysicsForceRequest(BaseModel):
    object_id: str
    force: List[float] = Field(default_factory=lambda: [0, 0, -50])


class PhysicsPushRequest(BaseModel):
    object_id: str
    direction: str = "forward"
    strength: float = 10.0


class PhysicsDropRequest(BaseModel):
    object_id: str
    height: float = 2.0


class PhysicsRotateRequest(BaseModel):
    object_id: str
    axis: str = "y"
    degrees: float = 45.0


@app.post("/physics/init")
async def physics_init():
    """Create physics bodies from current scene objects."""
    objects = [obj.to_dict() for obj in axis_state.semantic_map.objects.values()]
    if not objects:
        return {"status": "empty", "bodies": []}
    state = axis_state.physics.init_from_objects(objects)
    axis_state._append_event(f"Physics: {len(objects)} bodies initialized", "system")
    return state


@app.get("/physics/state")
async def physics_state():
    """Get current physics simulation state."""
    return axis_state.physics.get_state()


@app.post("/physics/step")
async def physics_step():
    """Run a single physics tick (1/60s)."""
    result = axis_state.physics.step()
    return result


@app.post("/physics/step_multi")
async def physics_step_multi(steps: int = 10):
    """Run multiple physics ticks."""
    for _ in range(min(steps, 120)):
        axis_state.physics.step()
    return axis_state.physics.get_state()


@app.post("/physics/apply_force")
async def physics_apply_force(payload: PhysicsForceRequest):
    """Apply a continuous force to a body."""
    result = axis_state.physics.apply_force_to(payload.object_id, payload.force)
    if result is None:
        return {"error": "body not found"}
    return result


@app.post("/physics/push")
async def physics_push(payload: PhysicsPushRequest):
    """Push a body in a direction."""
    result = axis_state.physics.push(payload.object_id, payload.direction, payload.strength)
    if result is None:
        return {"error": "body not found"}
    return result


@app.post("/physics/drop")
async def physics_drop(payload: PhysicsDropRequest):
    """Drop a body from a height."""
    result = axis_state.physics.drop(payload.object_id, payload.height)
    if result is None:
        return {"error": "body not found"}
    return result


@app.post("/physics/rotate")
async def physics_rotate(payload: PhysicsRotateRequest):
    """Apply rotational impulse to a body."""
    result = axis_state.physics.rotate(payload.object_id, payload.axis, payload.degrees)
    if result is None:
        return {"error": "body not found"}
    return result


@app.post("/physics/reset")
async def physics_reset():
    """Reset the physics simulation."""
    axis_state.physics.reset()
    axis_state._append_event("Physics simulation reset", "system")
    return axis_state.physics.get_state()


@app.get("/physics/presets")
async def physics_presets():
    """Get named force presets for UI."""
    return sim_force_vectors()


@app.get("/scene/intelligence")
async def get_scene_intelligence():
    """Get scene graph intelligence data (events, interactions, person tree)."""
    objects_list = [obj.to_dict() for obj in axis_state.semantic_map.objects.values()]
    intel = axis_state.intelligence.analyze_frame(
        objects_list, axis_state.human_poses, axis_state.scene_graph
    )
    return {
        **intel,
        "person_tree": axis_state.intelligence.build_person_tree(axis_state.scene_graph),
        "scene_graph": axis_state.scene_graph.to_dict(),
    }


class AgentQueryRequest(BaseModel):
    question: str
    use_llm: bool = True


@app.post("/agent/query")
async def agent_query(payload: AgentQueryRequest):
    """Ask the reasoning agent a question about the scene."""
    if axis_state.langgraph_agent and payload.use_llm:
        ctx = axis_state._build_scene_context()
        result = axis_state.langgraph_agent.query(
            question=payload.question,
            scene_context=ctx,
        )
        return {
            "response": result.response,
            "confidence": result.confidence,
            "sources": result.context_used,
            "langgraph": True,
        }
    result = axis_state.agent.query(payload.question)
    result["langgraph"] = False
    return result


@app.post("/agent/summarize")
async def agent_summarize():
    """Get a summary of recent activity."""
    if axis_state.langgraph_agent:
        ctx = axis_state._build_scene_context()
        summary = axis_state.langgraph_agent.summarize(ctx)
        return {"summary": summary, "langgraph": True}
    return axis_state.agent.summarize()


@app.post("/agent/suggest")
async def agent_suggest():
    """Get action recommendations."""
    if axis_state.langgraph_agent:
        ctx = axis_state._build_scene_context()
        recs = axis_state.langgraph_agent.recommend_actions(ctx)
        return {"recommendations": " ".join(recs) if recs else "No recommendations available.", "langgraph": True}
    return axis_state.agent.suggest()


@app.get("/agent/status")
async def agent_status():
    """Get agent status."""
    base = axis_state.agent.get_status()
    if axis_state.langgraph_agent:
        base["langgraph_model"] = axis_state.langgraph_agent.model_name
    return base


@app.post("/agent/reset")
async def agent_reset():
    """Reset conversation history."""
    axis_state.agent.reset_conversation()
    if axis_state.langgraph_agent:
        axis_state.langgraph_agent = None
        axis_state._init_langgraph()
    return {"status": "reset"}


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
