"""Scene Graph Intelligence — converts observations into symbolic knowledge.

Analyzes object trajectories, human poses, and detection history to
extract interactions, events, and enriched relationships for the scene graph.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..core.types import Vector3
from .scene_graph import SceneGraph, RelationType


@dataclass
class SymbolicEvent:
    """A high-level symbolic event extracted from observations."""
    event_id: str = ""
    event_type: str = ""       # picked_up, put_down, entered, exited, moved, approached, walked_away
    subject: str = ""          # person or object id
    subject_class: str = ""
    object: str = ""           # target object id (if applicable)
    object_class: str = ""
    confidence: float = 0.0
    timestamp: float = 0.0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "subject": self.subject,
            "subject_class": self.subject_class,
            "object": self.object,
            "object_class": self.object_class,
            "confidence": round(self.confidence, 3),
            "timestamp": self.timestamp,
            "description": self.description,
        }


@dataclass
class Interaction:
    """An interaction relationship between two entities."""
    relation_type: RelationType
    source_id: str
    source_label: str
    target_id: str
    target_label: str
    confidence: float
    last_observed: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation": self.relation_type.value,
            "source_id": self.source_id,
            "source_label": self.source_label,
            "target_id": self.target_id,
            "target_label": self.target_label,
            "confidence": round(self.confidence, 3),
            "last_observed": self.last_observed,
            "metadata": self.metadata,
        }


class InteractionAnalyzer:
    """Detects human-object interactions from pose + object data."""

    def __init__(self) -> None:
        self._previous_objects: Dict[str, Dict[str, Any]] = {}
        self._tracked_interactions: Dict[str, Interaction] = {}
        self._last_person_pos: Optional[Dict[str, float]] = None

    def analyze(
        self,
        objects: List[Dict[str, Any]],
        human_poses: List[Dict[str, Any]],
    ) -> Tuple[List[SymbolicEvent], List[Interaction]]:
        """Analyze current frame and return new events + active interactions."""
        events: List[SymbolicEvent] = []
        interactions: List[Interaction] = []
        now = time.time()

        current_ids = set()
        person_objects = [o for o in objects if o.get("class_name", "").lower() == "person"]
        other_objects = [o for o in objects if o.get("class_name", "").lower() != "person"]

        # Process person objects for entered events
        for p_obj in person_objects:
            oid = p_obj.get("object_id", "")
            if not oid:
                continue
            current_ids.add(oid)
            if oid not in self._previous_objects:
                events.append(SymbolicEvent(
                    event_id=f"entered_{oid}_{now}",
                    event_type="entered",
                    subject=oid,
                    subject_class="person",
                    description=f"person entered the scene",
                    confidence=0.8,
                    timestamp=now,
                ))

        # Process other objects for entered/moved/interaction events
        person_id = person_objects[0].get("object_id") if person_objects else None
        person_pos = person_objects[0].get("position", {}) if person_objects else None

        for obj in other_objects:
            oid = obj.get("object_id", "")
            if not oid:
                continue
            current_ids.add(oid)
            cls_name = obj.get("class_name", "unknown")
            pos = obj.get("position", {})

            # Detect entered (new object)
            if oid not in self._previous_objects:
                events.append(SymbolicEvent(
                    event_id=f"entered_{oid}_{now}",
                    event_type="entered",
                    subject=oid,
                    subject_class=cls_name,
                    description=f"{cls_name} entered the scene",
                    confidence=0.7,
                    timestamp=now,
                ))

            # Detect movement
            prev = self._previous_objects.get(oid)
            if prev:
                prev_pos = prev.get("position", {})
                dx = pos.get("x", 0) - prev_pos.get("x", 0)
                dy = pos.get("y", 0) - prev_pos.get("y", 0)
                dz = pos.get("z", 0) - prev_pos.get("z", 0)
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)

                if dist > 0.35:
                    events.append(SymbolicEvent(
                        event_id=f"moved_{oid}_{now}",
                        event_type="moved",
                        subject=oid,
                        subject_class=cls_name,
                        description=f"{cls_name} moved {dist:.2f}m",
                        confidence=min(0.9, 0.5 + dist),
                        timestamp=now,
                    ))

                # Detect pick_up / put_down:
                # If person is nearby and object appeared/disappeared
                if person_pos and person_id:
                    pdx = pos.get("x", 0) - person_pos.get("x", 0)
                    pdy = pos.get("y", 0) - person_pos.get("y", 0)
                    pdz = pos.get("z", 0) - person_pos.get("z", 0)
                    person_dist = math.sqrt(pdx * pdx + pdy * pdy + pdz * pdz)

                    # Pre-check: person_near object
                    prev_person_pos = self._last_person_pos
                    if prev_person_pos and person_dist < 0.8:
                        # Object moving toward person center = picked_up
                        if dist > 0.2 and abs(dy) > 0.1:
                            events.append(SymbolicEvent(
                                event_id=f"interact_{oid}_{now}",
                                event_type="interacts_with",
                                subject=person_id,
                                subject_class="person",
                                object=oid,
                                object_class=cls_name,
                                description=f"person interacts with {cls_name}",
                                confidence=0.6,
                                timestamp=now,
                            ))

        # Detect exited (removed objects)
        for oid, prev_obj in self._previous_objects.items():
            if oid not in current_ids and oid in self._previous_objects:
                cls_name = prev_obj.get("class_name", "unknown")
                events.append(SymbolicEvent(
                    event_id=f"exited_{oid}_{now}",
                    event_type="exited",
                    subject=oid,
                    subject_class=cls_name,
                    description=f"{cls_name} left the scene",
                    confidence=0.6,
                    timestamp=now,
                ))

        # Build interaction edges
        if person_pos and person_id:
            for obj in other_objects:
                oid = obj.get("object_id", "")
                cls_name = obj.get("class_name", "unknown")
                pos = obj.get("position", {})
                dx = pos.get("x", 0) - person_pos.get("x", 0)
                dy = pos.get("y", 0) - person_pos.get("y", 0)
                dz = pos.get("z", 0) - person_pos.get("z", 0)
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)

                if dist < 1.8:
                    interactions.append(Interaction(
                        relation_type=RelationType.NEAR,
                        source_id=person_id,
                        source_label="person",
                        target_id=oid,
                        target_label=cls_name,
                        confidence=max(0.4, 1.0 - dist / 2.5),
                        last_observed=now,
                        metadata={"distance": round(dist, 3)},
                    ))

                if dist < 0.8:
                    interactions.append(Interaction(
                        relation_type=RelationType.INTERACTS_WITH,
                        source_id=person_id,
                        source_label="person",
                        target_id=oid,
                        target_label=cls_name,
                        confidence=max(0.5, 1.0 - dist * 0.5),
                        last_observed=now,
                        metadata={"distance": round(dist, 3)},
                    ))

                # Look direction estimation (simplified: person above and facing the object)
                if abs(dx) < 1.5 and dy > -0.3 and dist < 2.5:
                    interactions.append(Interaction(
                        relation_type=RelationType.LOOKS_AT,
                        source_id=person_id,
                        source_label="person",
                        target_id=oid,
                        target_label=cls_name,
                        confidence=round(max(0.3, 0.7 - dist * 0.2), 3),
                        last_observed=now,
                    ))

        self._previous_objects = {
            oid: {"position": dict(obj.get("position", {})), "class_name": obj.get("class_name", "")}
            for oid, obj in [(o.get("object_id", ""), o) for o in objects if o.get("object_id")]
        }

        if person_pos:
            self._last_person_pos = dict(person_pos)

        return events, interactions


class MotionAnalyzer:
    """Analyzes object movement patterns and trajectories."""

    @staticmethod
    def classify_motion(obj: Dict[str, Any]) -> str:
        """Classify object motion type."""
        pos = obj.get("position", {})
        vel = obj.get("velocity", {})
        speed = math.sqrt(
            vel.get("x", 0) ** 2 + vel.get("y", 0) ** 2 + vel.get("z", 0) ** 2
        )
        if speed < 0.05:
            return "stationary"
        if speed > 2.0:
            return "fast"
        if abs(vel.get("y", 0)) > abs(vel.get("x", 0)) and abs(vel.get("y", 0)) > abs(vel.get("z", 0)):
            return "vertical"
        return "moving"


class SceneIntelligence:
    """Orchestrates scene graph intelligence — event tracking and relationship extraction.

    Converts raw scene observations into symbolic events and enriched relationships,
    integrating them into the scene graph.
    """

    def __init__(self) -> None:
        self.interaction_analyzer = InteractionAnalyzer()
        self.motion_analyzer = MotionAnalyzer()
        self.events: List[SymbolicEvent] = []
        self.interactions: Dict[str, Interaction] = {}
        self._event_counter = 0

    def analyze_frame(
        self,
        objects: List[Dict[str, Any]],
        human_poses: List[Dict[str, Any]],
        existing_graph: SceneGraph,
    ) -> Dict[str, Any]:
        """Analyze current frame data and return intelligence results."""
        new_events, new_interactions = self.interaction_analyzer.analyze(objects, human_poses)

        # Add new events
        for ev in new_events:
            self._event_counter += 1
            ev.event_id = f"ev_{self._event_counter}"
            self.events.append(ev)
        self.events = self.events[-50:]

        # Update interaction registry
        for inter in new_interactions:
            key = f"{inter.source_id}_{inter.relation_type.value}_{inter.target_id}"
            existing = self.interactions.get(key)
            if existing:
                existing.confidence = max(existing.confidence, inter.confidence)
                existing.last_observed = inter.last_observed
                existing.metadata = inter.metadata
            else:
                self.interactions[key] = inter

        # Enrich scene graph with interaction edges
        self._enrich_graph(existing_graph)

        return {
            "events": [e.to_dict() for e in self.events[-20:]],
            "active_interactions": [i.to_dict() for i in self.interactions.values()],
            "event_count": len(self.events),
            "interaction_count": len(self.interactions),
        }

    def _enrich_graph(self, graph: SceneGraph) -> None:
        """Add intelligence-derived edges to the scene graph."""
        for inter in self.interactions.values():
            # Ensure nodes exist
            if inter.source_id not in graph.nodes or inter.target_id not in graph.nodes:
                continue

            # Add edge if it doesn't already exist
            existing = False
            for edge in graph.edges.values():
                if (edge.source_id == inter.source_id
                    and edge.target_id == inter.target_id
                    and edge.relation_type == inter.relation_type):
                    existing = True
                    edge.update_observation(inter.confidence)
                    break

            if not existing:
                graph.add_edge(
                    source_id=inter.source_id,
                    target_id=inter.target_id,
                    relation=inter.relation_type,
                    confidence=inter.confidence,
                )

    def build_person_tree(self, graph: SceneGraph) -> Optional[Dict[str, Any]]:
        """Build a person-centric tree view of the scene graph."""
        # Find the person node
        person_node = None
        for nid, node in graph.nodes.items():
            if node.label.lower() == "person" and node.node_type == "object":
                person_node = nid
                break

        if person_node is None:
            return None

        tree = graph.traverse(start_id=person_node, max_depth=2)
        return tree

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_events": len(self.events),
            "active_interactions": len(self.interactions),
            "interaction_types": defaultdict(int),
        }

    def reset(self) -> None:
        self.events.clear()
        self.interactions.clear()
        self._event_counter = 0
        self.interaction_analyzer._previous_objects.clear()
        self.interaction_analyzer._tracked_interactions.clear()
        self.interaction_analyzer._last_person_pos = None
