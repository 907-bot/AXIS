"""Scene graph for representing object relationships."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from enum import Enum
from datetime import datetime
import numpy as np
from loguru import logger

from ..core.types import Vector3


class RelationType(Enum):
    """Types of spatial/ semantic relationships."""
    # Spatial
    NEAR = "near"
    FAR = "far"
    ON = "on"
    UNDER = "under"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    INSIDE = "inside"
    CONTAINS = "contains"
    
    # Semantic
    HOLDS = "holds"
    LOOKS_AT = "looks_at"
    FACING = "facing"
    FACING_AWAY = "facing_away"
    
    # Actions
    INTERACTS_WITH = "interacts_with"
    DEPENDS_ON = "depends_on"


@dataclass
class SceneNode:
    """Node in scene graph representing an entity."""
    node_id: str
    label: str
    node_type: str  # "object", "person", "location", "event"
    
    # Properties
    position: Optional[Vector3] = None
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Relationships
    outgoing_edges: Set[str] = field(default_factory=set)
    incoming_edges: Set[str] = field(default_factory=set)
    
    # Temporal
    created_at: float = field(default_factory=datetime.now().timestamp)
    updated_at: float = field(default_factory=datetime.now().timestamp)

    def add_property(self, key: str, value: Any):
        """Add/update property."""
        self.properties[key] = value
        self.updated_at = datetime.now().timestamp()

    def get_property(self, key: str, default: Any = None) -> Any:
        """Get property value."""
        return self.properties.get(key, default)


@dataclass
class SceneEdge:
    """Edge in scene graph representing a relationship."""
    edge_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    
    # Confidence and temporal info
    confidence: float = 1.0
    first_observed: float = field(default_factory=datetime.now().timestamp)
    last_observed: float = field(default_factory=datetime.now().timestamp)
    
    # Optional properties
    properties: Dict[str, Any] = field(default_factory=dict)

    def update_observation(self, confidence: float):
        """Update edge with new observation."""
        self.confidence = max(self.confidence, confidence)
        self.last_observed = datetime.now().timestamp()


class SceneGraph:
    """
    Scene graph for representing spatial and semantic relationships.
    
    Provides:
    - Node management (objects, persons, locations)
    - Edge management (relationships)
    - Graph traversal and queries
    - Temporal reasoning
    """

    def __init__(self):
        self.nodes: Dict[str, SceneNode] = {}
        self.edges: Dict[str, SceneEdge] = {}
        self._edge_index: Dict[Tuple[str, str, RelationType], str] = {}

    def add_node(
        self,
        node_id: str,
        label: str,
        node_type: str = "object",
        position: Optional[Vector3] = None,
        **properties
    ) -> SceneNode:
        """Add node to graph."""
        node = SceneNode(
            node_id=node_id,
            label=label,
            node_type=node_type,
            position=position,
            properties=properties
        )
        self.nodes[node_id] = node
        logger.debug(f"Added node {node_id} ({label})")
        return node

    def get_node(self, node_id: str) -> Optional[SceneNode]:
        """Get node by ID."""
        return self.nodes.get(node_id)

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: RelationType,
        confidence: float = 1.0,
        **properties
    ) -> Optional[SceneEdge]:
        """Add edge to graph."""
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.warning(f"Cannot add edge: node not found")
            return None

        edge_id = f"{source_id}_{relation.value}_{target_id}"

        edge = SceneEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation,
            confidence=confidence,
            properties=properties
        )

        self.edges[edge_id] = edge
        self.nodes[source_id].outgoing_edges.add(edge_id)
        self.nodes[target_id].incoming_edges.add(edge_id)
        self._edge_index[(source_id, target_id, relation)] = edge_id

        logger.debug(f"Added edge: {source_id} --[{relation.value}]--> {target_id}")
        return edge

    def get_edge(self, edge_id: str) -> Optional[SceneEdge]:
        """Get edge by ID."""
        return self.edges.get(edge_id)

    def get_edges_between(self, source_id: str, target_id: str) -> List[SceneEdge]:
        """Get all edges between two nodes."""
        return [
            self.edges[eid]
            for eid in self.nodes.get(source_id, SceneNode("", "", "")).outgoing_edges
            if self.edges[eid].target_id == target_id
        ]

    def get_relations_of_type(
        self,
        source_id: Optional[str] = None,
        relation: Optional[RelationType] = None
    ) -> List[SceneEdge]:
        """Get edges matching criteria."""
        results = []
        for edge in self.edges.values():
            if source_id and edge.source_id != source_id:
                continue
            if relation and edge.relation_type != relation:
                continue
            results.append(edge)
        return results

    def traverse(
        self,
        start_id: str,
        max_depth: int = 3,
        relation_filter: Optional[List[RelationType]] = None
    ) -> Dict[str, Any]:
        """
        Traverse graph from start node.
        
        Returns hierarchical structure of reachable nodes.
        """
        visited = set()
        result = {"id": start_id, "label": self.nodes[start_id].label if start_id in self.nodes else None, "children": []}
        
        def _traverse(node_id: str, depth: int, parent: Dict):
            if depth > max_depth or node_id in visited:
                return
            
            visited.add(node_id)
            node = self.nodes.get(node_id)
            if node is None:
                return
            
            for edge_id in node.outgoing_edges:
                edge = self.edges.get(edge_id)
                if edge is None:
                    continue
                
                if relation_filter and edge.relation_type not in relation_filter:
                    continue
                
                child_node = self.nodes.get(edge.target_id)
                if child_node:
                    child_dict = {
                        "id": edge.target_id,
                        "label": child_node.label,
                        "relation": edge.relation_type.value,
                        "confidence": edge.confidence,
                        "children": []
                    }
                    parent["children"].append(child_dict)
                    _traverse(edge.target_id, depth + 1, child_dict)

        _traverse(start_id, 0, result)
        return result

    def find_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        """Find shortest path between two nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        if source_id == target_id:
            return [source_id]

        # BFS
        queue = [(source_id, [source_id])]
        visited = {source_id}

        while queue:
            current, path = queue.pop(0)
            
            for edge_id in self.nodes[current].outgoing_edges:
                edge = self.edges.get(edge_id)
                if edge and edge.target_id not in visited:
                    new_path = path + [edge.target_id]
                    
                    if edge.target_id == target_id:
                        return new_path
                    
                    visited.add(edge.target_id)
                    queue.append((edge.target_id, new_path))

        return None

    def get_subgraph(
        self,
        center_id: str,
        radius: int = 2
    ) -> "SceneGraph":
        """Extract subgraph around center node."""
        visited = {center_id}
        node_ids = {center_id}
        
        for _ in range(radius):
            new_nodes = set()
            for nid in visited:
                node = self.nodes.get(nid)
                if node:
                    for edge_id in node.outgoing_edges:
                        edge = self.edges.get(edge_id)
                        if edge:
                            new_nodes.add(edge.target_id)
            
            visited.update(new_nodes)
            node_ids.update(new_nodes)

        subgraph = SceneGraph()
        for nid in node_ids:
            node = self.nodes.get(nid)
            if node:
                subgraph.nodes[nid] = node

        for edge in self.edges.values():
            if edge.source_id in node_ids and edge.target_id in node_ids:
                subgraph.edges[edge.edge_id] = edge

        return subgraph

    def remove_node(self, node_id: str) -> bool:
        """Remove node and its edges."""
        if node_id not in self.nodes:
            return False

        # Remove all edges
        for edge_id in list(self.nodes[node_id].outgoing_edges):
            self.remove_edge(edge_id)
        for edge_id in list(self.nodes[node_id].incoming_edges):
            self.remove_edge(edge_id)

        del self.nodes[node_id]
        return True

    def remove_edge(self, edge_id: str) -> bool:
        """Remove edge."""
        if edge_id not in self.edges:
            return False

        edge = self.edges[edge_id]
        if edge.source_id in self.nodes:
            self.nodes[edge.source_id].outgoing_edges.discard(edge_id)
        if edge.target_id in self.nodes:
            self.nodes[edge.target_id].incoming_edges.discard(edge_id)

        del self.edges[edge_id]
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Export graph as dictionary."""
        return {
            "nodes": {
                nid: {
                    "id": node.node_id,
                    "label": node.label,
                    "type": node.node_type,
                    "position": {
                        "x": node.position.x,
                        "y": node.position.y,
                        "z": node.position.z
                    } if node.position else None,
                    "properties": node.properties
                }
                for nid, node in self.nodes.items()
            },
            "edges": [
                {
                    "id": edge.edge_id,
                    "source": edge.source_id,
                    "target": edge.target_id,
                    "relation": edge.relation_type.value,
                    "confidence": edge.confidence
                }
                for edge in self.edges.values()
            ]
        }

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return len(self.edges)