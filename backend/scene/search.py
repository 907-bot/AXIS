"""Semantic search using vector embeddings."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from loguru import logger

from .semantic_map import SemanticMap, MapObject


@dataclass
class SearchResult:
    """Semantic search result."""
    object_id: str
    class_name: str
    position: Tuple[float, float, float]
    similarity: float
    confidence: float
    distance: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "class_name": self.class_name,
            "position": {"x": self.position[0], "y": self.position[1], "z": self.position[2]},
            "similarity": self.similarity,
            "confidence": self.confidence,
            "distance": self.distance
        }


class SemanticSearch:
    """
    Semantic search engine for 3D scene.
    
    Provides:
    - Text-to-scene search (e.g., "find my bottle")
    - Image-to-scene search
    - Spatial queries
    - Hybrid search with filters
    """

    def __init__(self, semantic_map: SemanticMap):
        self.semantic_map = semantic_map
        self.embedding_model = None

    def set_embedding_model(self, model):
        """Set embedding model for text/image queries."""
        self.embedding_model = model

    def search_by_text(
        self,
        query: str,
        top_k: int = 5,
        class_filter: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Search objects by text query."""
        if self.embedding_model is None:
            logger.warning("No embedding model configured")
            return []

        # Get query embedding
        query_embedding = self.embedding_model.extract_text(query)

        # Search
        results = self.semantic_map.semantic_search(
            query_embedding,
            top_k=top_k * 2 if class_filter else top_k,
            use_clip=True
        )

        search_results = []
        for obj, similarity in results:
            if class_filter and obj.class_name not in class_filter:
                continue

            search_results.append(SearchResult(
                object_id=obj.object_id,
                class_name=obj.class_name,
                position=(obj.position.x, obj.position.y, obj.position.z),
                similarity=float(similarity),
                confidence=obj.confidence
            ))

            if len(search_results) >= top_k:
                break

        return search_results

    def search_by_image(
        self,
        image: np.ndarray,
        top_k: int = 5
    ) -> List[SearchResult]:
        """Search objects by image similarity."""
        if self.embedding_model is None:
            return []

        query_embedding = self.embedding_model.extract_image(image)

        results = self.semantic_map.semantic_search(
            query_embedding,
            top_k=top_k,
            use_clip=True
        )

        return [
            SearchResult(
                object_id=obj.object_id,
                class_name=obj.class_name,
                position=(obj.position.x, obj.position.y, obj.position.z),
                similarity=float(sim),
                confidence=obj.confidence
            )
            for obj, sim in results
        ]

    def search_by_location(
        self,
        position: Tuple[float, float, float],
        radius: float = 1.0,
        class_filter: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Search objects by spatial proximity."""
        from ..core.types import Vector3
        
        center = Vector3(x=position[0], y=position[1], z=position[2])
        objects = self.semantic_map.get_objects_in_radius(center, radius)

        if class_filter:
            objects = [o for o in objects if o.class_name in class_filter]

        return [
            SearchResult(
                object_id=obj.object_id,
                class_name=obj.class_name,
                position=(obj.position.x, obj.position.y, obj.position.z),
                similarity=1.0,
                confidence=obj.confidence,
                distance=np.sqrt(
                    (obj.position.x - position[0])**2 +
                    (obj.position.y - position[1])**2 +
                    (obj.position.z - position[2])**2
                )
            )
            for obj in objects
        ]

    def hybrid_search(
        self,
        query: Optional[str] = None,
        image: Optional[np.ndarray] = None,
        position: Optional[Tuple[float, float, float]] = None,
        radius: float = 2.0,
        top_k: int = 5
    ) -> List[SearchResult]:
        """
        Combine multiple search modalities.
        
        Prioritizes:
        1. Spatial if position given
        2. Text/image semantic similarity
        """
        candidates: Dict[str, float] = {}

        # Spatial search
        if position:
            spatial_results = self.search_by_location(position, radius)
            for result in spatial_results:
                candidates[result.object_id] = max(
                    candidates.get(result.object_id, 0),
                    0.5 * (1 - (result.distance or 0) / radius)
                )

        # Text search
        if query:
            text_results = self.search_by_text(query, top_k=top_k * 2)
            for result in text_results:
                candidates[result.object_id] = max(
                    candidates.get(result.object_id, 0),
                    result.similarity
                )

        # Image search
        if image is not None:
            image_results = self.search_by_image(image, top_k=top_k * 2)
            for result in image_results:
                candidates[result.object_id] = max(
                    candidates.get(result.object_id, 0),
                    result.similarity
                )

        # Sort and return top k
        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

        results = []
        for obj_id, score in sorted_candidates[:top_k]:
            obj = self.semantic_map.get_object(obj_id)
            if obj:
                results.append(SearchResult(
                    object_id=obj.object_id,
                    class_name=obj.class_name,
                    position=(obj.position.x, obj.position.y, obj.position.z),
                    similarity=score,
                    confidence=obj.confidence
                ))

        return results

    def describe_result(self, result: SearchResult) -> str:
        """Generate natural language description of search result."""
        return f"{result.class_name} at position ({result.position[0]:.2f}, {result.position[1]:.2f}, {result.position[2]:.2f}) with {result.similarity*100:.0f}% confidence"