"""Semantic map for storing and querying 3D objects with embeddings."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np
from loguru import logger

from ..core.types import Vector3, BoundingBox3D


@dataclass
class MapObject:
    """Semantic object in the 3D map."""
    object_id: str
    class_name: str
    position: Vector3
    size: Vector3  # width, height, depth
    bounding_box: Optional[BoundingBox3D] = None
    
    # Semantic features
    clip_embedding: Optional[np.ndarray] = None
    dino_embedding: Optional[np.ndarray] = None
    color: Optional[Tuple[int, int, int]] = None
    
    # Metadata
    confidence: float = 1.0
    first_observed: float = field(default_factory=datetime.now().timestamp)
    last_observed: float = field(default_factory=datetime.now().timestamp)
    observation_count: int = 1
    
    # Tracking
    trajectory: List[Vector3] = field(default_factory=list)
    is_static: bool = True
    
    def update_position(self, new_position: Vector3, timestamp: float):
        """Update object position."""
        self.trajectory.append(new_position)
        self.position = new_position
        self.last_observed = timestamp
        self.observation_count += 1
        
        # Check if static (minimal movement)
        if len(self.trajectory) > 5:
            recent = self.trajectory[-5:]
            movements = [
                np.sqrt((p.x - q.x)**2 + (p.y - q.y)**2 + (p.z - q.z)**2)
                for p, q in zip(recent[:-1], recent[1:])
            ]
            avg_movement = np.mean(movements)
            self.is_static = avg_movement < 0.01  # 1cm threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "object_id": self.object_id,
            "class_name": self.class_name,
            "position": {"x": self.position.x, "y": self.position.y, "z": self.position.z},
            "size": {"x": self.size.x, "y": self.size.y, "z": self.size.z},
            "confidence": self.confidence,
            "first_observed": self.first_observed,
            "last_observed": self.last_observed,
            "observation_count": self.observation_count,
            "is_static": self.is_static,
            "trajectory_length": len(self.trajectory)
        }


class SemanticMap:
    """
    Semantic map storing 3D objects with semantic embeddings.
    
    Provides:
    - Object storage and retrieval
    - Semantic search via embeddings
    - Spatial queries
    - Object tracking over time
    """

    def __init__(self, max_objects: int = 10000):
        self.max_objects = max_objects
        self.objects: Dict[str, MapObject] = {}
        self.class_counts: Dict[str, int] = {}
        self._embedding_dim = 0

    def add_object(
        self,
        object_id: str,
        class_name: str,
        position: Vector3,
        size: Vector3,
        clip_embedding: Optional[np.ndarray] = None,
        dino_embedding: Optional[np.ndarray] = None,
        color: Optional[Tuple[int, int, int]] = None,
        confidence: float = 1.0
    ) -> MapObject:
        """Add new object to map."""
        obj = MapObject(
            object_id=object_id,
            class_name=class_name,
            position=position,
            size=size,
            clip_embedding=clip_embedding,
            dino_embedding=dino_embedding,
            color=color,
            confidence=confidence
        )

        self.objects[object_id] = obj
        self.class_counts[class_name] = self.class_counts.get(class_name, 0) + 1

        if clip_embedding is not None:
            self._embedding_dim = len(clip_embedding)

        logger.debug(f"Added object {object_id} ({class_name}) at ({position.x:.2f}, {position.y:.2f}, {position.z:.2f})")
        return obj

    def update_object(
        self,
        object_id: str,
        position: Optional[Vector3] = None,
        **kwargs
    ):
        """Update existing object."""
        if object_id not in self.objects:
            logger.warning(f"Object {object_id} not found")
            return

        obj = self.objects[object_id]
        
        if position is not None:
            obj.update_position(position, kwargs.get("timestamp", datetime.now().timestamp()))
        
        if "confidence" in kwargs:
            obj.confidence = kwargs["confidence"]
        
        if "clip_embedding" in kwargs:
            obj.clip_embedding = kwargs["clip_embedding"]
        if "dino_embedding" in kwargs:
            obj.dino_embedding = kwargs["dino_embedding"]

    def get_object(self, object_id: str) -> Optional[MapObject]:
        """Get object by ID."""
        return self.objects.get(object_id)

    def get_objects_in_radius(self, center: Vector3, radius: float) -> List[MapObject]:
        """Get objects within radius of center."""
        results = []
        for obj in self.objects.values():
            dx = obj.position.x - center.x
            dy = obj.position.y - center.y
            dz = obj.position.z - center.z
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            if dist <= radius:
                results.append((obj, dist))

        results.sort(key=lambda x: x[1])
        return [r[0] for r in results]

    def get_objects_by_class(self, class_name: str) -> List[MapObject]:
        """Get all objects of a class."""
        return [obj for obj in self.objects.values() if obj.class_name == class_name]

    def semantic_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        use_clip: bool = True
    ) -> List[Tuple[MapObject, float]]:
        """Search objects by semantic similarity."""
        results = []

        for obj in self.objects.values():
            if use_clip and obj.clip_embedding is not None:
                embedding = obj.clip_embedding
            elif obj.dino_embedding is not None:
                embedding = obj.dino_embedding
            else:
                continue

            # Cosine similarity
            sim = self._cosine_similarity(query_embedding, embedding)
            results.append((obj, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def text_search(self, query: str, embedding_model) -> List[Tuple[MapObject, float]]:
        """Search objects by text query."""
        query_embedding = embedding_model.extract_text(query)
        return self.semantic_search(query_embedding, use_clip=True)

    def remove_object(self, object_id: str) -> bool:
        """Remove object from map."""
        if object_id in self.objects:
            obj = self.objects[object_id]
            self.class_counts[obj.class_name] = max(0, self.class_counts[obj.class_name] - 1)
            del self.objects[object_id]
            return True
        return False

    def merge_nearby_objects(self, threshold: float = 0.1) -> int:
        """Merge objects that are very close together."""
        merged = 0
        to_merge: Dict[str, str] = {}

        obj_ids = list(self.objects.keys())
        for i, id1 in enumerate(obj_ids):
            for id2 in obj_ids[i+1:]:
                if id1 in to_merge or id2 in to_merge:
                    continue

                obj1 = self.objects[id1]
                obj2 = self.objects[id2]

                dx = obj1.position.x - obj2.position.x
                dy = obj1.position.y - obj2.position.y
                dz = obj1.position.z - obj2.position.z
                dist = np.sqrt(dx**2 + dy**2 + dz**2)

                if dist < threshold and obj1.class_name == obj2.class_name:
                    to_merge[id2] = id1

        for source, target in to_merge.items():
            self.remove_object(source)
            merged += 1

        return merged

    def get_stats(self) -> Dict[str, Any]:
        """Get map statistics."""
        return {
            "total_objects": len(self.objects),
            "class_counts": dict(self.class_counts),
            "static_objects": sum(1 for o in self.objects.values() if o.is_static),
            "dynamic_objects": sum(1 for o in self.objects.values() if not o.is_static),
            "total_observations": sum(o.observation_count for o in self.objects.values()),
            "embedding_dim": self._embedding_dim
        }

    def clear(self):
        """Clear all objects."""
        self.objects.clear()
        self.class_counts.clear()
        logger.info("Semantic map cleared")

    def to_dict(self) -> Dict[str, Any]:
        """Export map as dictionary."""
        return {
            "objects": {oid: obj.to_dict() for oid, obj in self.objects.items()},
            "stats": self.get_stats()
        }