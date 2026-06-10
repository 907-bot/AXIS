"""LLM Reasoning Agent for AXIS.

Collects context from all subsystems and generates intelligent responses
using OpenAI (when available) or a smart template-based fallback engine.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class AgentMessage:
    """A single message in the agent conversation history."""
    role: str          # "user" | "assistant" | "system"
    content: str
    timestamp: float = 0.0

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


SYSTEM_PROMPT = """You are AXIS, an embodied AI assistant that perceives the physical world through a webcam.

You have access to real-time data about:
- Objects detected in the 3D scene (class, position, confidence, motion status)
- Human pose and motion analytics (balance, stability, speed, joint angles)
- Scene graph relationships (spatial and interaction relationships between entities)
- Recent events (objects entering/exiting/moving, human interactions)
- Future predictions (object motion predictions, human action forecasts)
- Physics simulation state (active bodies, collisions)
- Scene statistics (object counts, detection history)

Rules:
1. Answer naturally based on the context provided. Do NOT mention the raw data format.
2. If no objects are detected, say you don't see anything yet and suggest starting the camera.
3. When asked "where is X", search for the object and describe its position relative to the person or other objects.
4. When asked "what happened" or "summarize", describe recent events chronologically.
5. When asked "what will happen next", use prediction data to forecast.
6. When asked for recommendations, suggest actions the user could take.
7. Be concise but descriptive. Use spatial language (left, right, near, far).
8. If the question is about something not in the scene data, say you don't have that information.

Respond in 1-3 sentences unless the user asks for detail."""


class ContextCollector:
    """Collects and formats context from all AXIS subsystems."""

    @staticmethod
    def collect(state_provider: Callable) -> Dict[str, Any]:
        """Gather full context from all subsystems."""
        state = state_provider()
        ctx = {}

        # Objects
        objects = state.get("objects", [])
        ctx["object_count"] = len(objects)
        ctx["objects"] = [
            {"class": o.get("class_name"), "confidence": o.get("confidence"),
             "position": o.get("position"), "static": o.get("is_static", True)}
            for o in objects[:15]
        ]

        # Scene stats
        stats = state.get("stats", {})
        ctx["class_counts"] = stats.get("class_counts", {})

        # Events
        events = state.get("recent_events", [])
        ctx["recent_events"] = [
            {"type": e.get("type"), "description": e.get("description"),
             "time": time.strftime("%H:%M:%S", time.localtime(e.get("timestamp", 0)))}
            for e in events[-10:]
        ]

        # Human
        poses = state.get("human_poses", [])
        analytics = state.get("human_analytics", {})
        ctx["human_present"] = len(poses) > 0
        if analytics:
            ctx["human_analytics"] = {
                "balance": analytics.get("balance"),
                "stability": analytics.get("stability"),
                "speed": analytics.get("velocity", {}).get("speed"),
                "is_moving": analytics.get("is_moving"),
                "joint_angles": analytics.get("joint_angles", {}),
            }

        # Prediction
        pred = state.get("prediction", {})
        ctx["prediction"] = {
            "action": pred.get("action", {}).get("action"),
            "action_confidence": pred.get("action", {}).get("confidence"),
            "tracked_objects": pred.get("stats", {}).get("tracked_objects"),
        }

        # Scene graph
        sg = state.get("scene_graph", {})
        ctx["graph_nodes"] = len(sg.get("nodes", {}))
        ctx["graph_edges"] = len(sg.get("edges", []))

        # Intelligence
        intel = state.get("intelligence", {})
        ctx["intel_events"] = intel.get("event_count", 0)
        ctx["intel_interactions"] = [
            {"source": i.get("source_label"), "relation": i.get("relation"),
             "target": i.get("target_label"), "confidence": i.get("confidence")}
            for i in intel.get("interactions", [])[:8]
        ]

        # Physics
        physics = state.get("physics", {})
        ctx["physics_bodies"] = physics.get("body_count", 0)
        ctx["physics_collisions"] = physics.get("total_collisions", 0)

        # Camera
        ctx["camera_running"] = state.get("camera_running", False)

        return ctx

    @staticmethod
    def format_for_prompt(ctx: Dict[str, Any]) -> str:
        """Format context into a readable text block for the LLM."""
        lines = ["## Current Scene State", ""]

        if not ctx["objects"]:
            lines.append("No objects currently detected.")
        else:
            lines.append(f"Objects detected ({ctx['object_count']}):")
            for o in ctx["objects"][:10]:
                p = o.get("position", {})
                status = "static" if o.get("static") else "moving"
                lines.append(f"  - {o['class']} (confidence: {o['confidence']:.0%}, {status}) at x={p.get('x',0):.2f}, y={p.get('y',0):.2f}, z={p.get('z',0):.2f}")

        lines.append("")
        if ctx["human_present"]:
            ha = ctx.get("human_analytics", {})
            lines.append(f"Human detected:")
            lines.append(f"  - Balance: {ha.get('balance', 'N/A')}, Stability: {ha.get('stability', 'N/A')}")
            lines.append(f"  - Speed: {ha.get('speed', 'N/A')}, Moving: {ha.get('is_moving', False)}")
        else:
            lines.append("No human detected.")

        if ctx["recent_events"]:
            lines.append("")
            lines.append("Recent events:")
            for e in ctx["recent_events"][-6:]:
                lines.append(f"  [{e['time']}] {e['description']}")

        pred = ctx.get("prediction", {})
        if pred.get("action"):
            lines.append("")
            lines.append(f"Predicted action: {pred['action']} (confidence: {pred.get('action_confidence', 0):.0%})")

        if ctx.get("intel_interactions"):
            lines.append("")
            lines.append("Relationships:")
            for i in ctx["intel_interactions"][:5]:
                lines.append(f"  {i['source']} --{i['relation']}--> {i['target']} ({i.get('confidence', 0):.0%})")

        lines.append("")
        lines.append(f"Scene graph: {ctx.get('graph_nodes', 0)} nodes, {ctx.get('graph_edges', 0)} edges")
        lines.append(f"Physics: {ctx.get('physics_bodies', 0)} bodies, {ctx.get('physics_collisions', 0)} collisions")
        lines.append(f"Camera: {'active' if ctx.get('camera_running') else 'inactive'}")

        return "\n".join(lines)


class SmartFallback:
    """Template-based reasoning engine when no LLM is available."""

    def __init__(self) -> None:
        pass

    def answer(self, question: str, ctx: Dict[str, Any]) -> str:
        lowered = question.lower().strip()

        # Empty scene
        if not ctx["objects"]:
            return "I don't see any objects yet. Please start the camera and point it around the room."

        # Where questions
        if "where" in lowered:
            return self._answer_where(question, ctx)

        # How many
        if "how many" in lowered:
            return self._answer_count(question, ctx)

        # What happened / summarize / recent
        if any(w in lowered for w in ["what happened", "summarize", "recent", "what changed"]):
            return self._answer_recent(ctx)

        # What will happen next / predict
        if any(w in lowered for w in ["predict", "next", "will happen", "forecast"]):
            return self._answer_predict(ctx)

        # What do you see / describe scene
        if any(w in lowered for w in ["what do you see", "describe", "scene", "what's happening"]):
            return self._answer_describe(ctx)

        # Recommend / what should I do
        if any(w in lowered for w in ["recommend", "suggest", "what should i do", "what can i do"]):
            return self._answer_recommend(ctx)

        # Default — give a response based on what we know
        return self._answer_general(ctx)

    def _answer_where(self, question: str, ctx: Dict[str, Any]) -> str:
        question_lower = question.lower()
        for obj in ctx["objects"]:
            cls = obj["class"].lower()
            if cls in question_lower:
                p = obj.get("position", {})
                return f"I can see the {obj['class']} at the center of the scene, slightly to the {'left' if p.get('x',0) < 0 else 'right'}, about {abs(p.get('z', 2)):.1f} meters away."
        return f"I don't see that object right now. I currently detect: {', '.join(o['class'] for o in ctx['objects'][:5])}."

    def _answer_count(self, question: str, ctx: Dict[str, Any]) -> str:
        counts = ctx.get("class_counts", {})
        for cls_name, count in counts.items():
            if cls_name.lower() in question.lower():
                return f"I see {count} {cls_name}{'s' if count != 1 else ''} in the scene right now."
        total = ctx["object_count"]
        return f"I currently detect {total} object{'s' if total != 1 else ''} total."

    def _answer_recent(self, ctx: Dict[str, Any]) -> str:
        events = ctx.get("recent_events", [])
        if not events:
            return "Nothing notable has happened recently. The scene has been stable."

        descriptions = [e["description"] for e in events[-5:]]
        combined = "; ".join(descriptions)
        return f"Here's what happened recently: {combined}."

    def _answer_predict(self, ctx: Dict[str, Any]) -> str:
        pred = ctx.get("prediction", {})
        action = pred.get("action")
        if action and action != "stationary":
            return f"I predict the person will keep {action} over the next few seconds."
        # Check moving objects
        moving = [o for o in ctx["objects"] if not o.get("static", True)]
        if moving:
            return f"I expect {moving[0]['class']} to continue moving through the scene."
        return "The scene appears stable. I expect things to stay in place unless something triggers movement."

    def _answer_describe(self, ctx: Dict[str, Any]) -> str:
        human = "a person" if ctx.get("human_present") else "no person"
        obj_list = ", ".join(o["class"] for o in ctx["objects"][:7])
        counts = ctx.get("class_counts", {})
        summary = ", ".join(f"{c} {l}" for l, c in counts.items())
        interactions = ctx.get("intel_interactions", [])
        rel_summary = ""
        if interactions:
            rel_summary = f" Relationships: " + "; ".join(
                f"{i['source']} is {i['relation']} {i['target']}" for i in interactions[:3])
        return f"I'm observing {human} with {ctx['object_count']} object{'s' if ctx['object_count'] != 1 else ''}: {summary}.{rel_summary}"

    def _answer_recommend(self, ctx: Dict[str, Any]) -> str:
        suggestions = []
        if not ctx["camera_running"]:
            suggestions.append("Start the camera so I can see what's in the room.")
        if not ctx["objects"]:
            suggestions.append("Point the camera at objects around the room so I can map them.")
        if ctx["human_present"]:
            ha = ctx.get("human_analytics", {})
            if ha.get("balance", 1) < 0.5:
                suggestions.append("Your balance seems unstable, consider sitting down.")
            if ha.get("speed", 0) > 5:
                suggestions.append("You're moving quickly — try slowing down for better tracking.")
        if ctx.get("physics_bodies", 0) > 0:
            suggestions.append("Try the Physics Sandbox — select an object and push it around!")
        if not suggestions:
            suggestions.append("The scene looks good! Try asking me about specific objects or events.")
        return " ".join(suggestions)

    def _answer_general(self, ctx: Dict[str, Any]) -> str:
        obj_list = ", ".join(o["class"] for o in ctx["objects"][:5])
        return f"I'm tracking {ctx['object_count']} object{'s' if ctx['object_count'] != 1 else ''}: {obj_list}. You can ask me where things are, what happened recently, or what I predict will happen next."

    def summarize(self, ctx: Dict[str, Any]) -> str:
        """Generate a summary of recent activity."""
        events = ctx.get("recent_events", [])
        human = "A person" if ctx.get("human_present") else "No one"
        obj_summary = ", ".join(f"{c} {l}" for l, c in ctx.get("class_counts", {}).items()) or "no objects"
        event_desc = ". ".join(e["description"] for e in events[-5:]) if events else "nothing notable"
        interactions = ctx.get("intel_interactions", [])
        rel_desc = ""
        if interactions:
            rel_desc = " " + "; ".join(
                f"{i['source']} {i['relation']} {i['target']}" for i in interactions[:3])
        return f"In the last few minutes, {human} was present with {obj_summary}. Events: {event_desc}.{rel_desc}"


class ReasoningAgent:
    """Main reasoning agent — uses OpenAI API or smart fallback."""

    def __init__(self, state_provider: Callable) -> None:
        self.state_provider = state_provider
        self.collector = ContextCollector()
        self.fallback = SmartFallback()
        self.messages: List[AgentMessage] = []
        self.max_history = 20

        # Try to import openai
        self.openai_client = None
        self.openai_model = os.environ.get("AXIS_LLM_MODEL", "gpt-4o-mini")
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if api_key:
            try:
                from openai import OpenAI
                self.openai_client = OpenAI(api_key=api_key, timeout=30.0)
                logger.info(f"OpenAI client initialized (model: {self.openai_model})")
            except ImportError:
                logger.warning("openai package not installed, using fallback")
            except Exception as e:
                logger.warning(f"OpenAI init failed: {e}, using fallback")

    @property
    def has_llm(self) -> bool:
        return self.openai_client is not None

    def query(self, question: str) -> Dict[str, Any]:
        """Ask a question and get a reasoned response."""
        ctx = self.collector.collect(self.state_provider)
        context_text = self.collector.format_for_prompt(ctx)

        # Add to history
        self.messages.append(AgentMessage(role="user", content=question, timestamp=time.time()))

        if self.openai_client:
            response_text, sources = self._query_llm(question, context_text)
        else:
            response_text = self.fallback.answer(question, ctx)
            sources = self._extract_sources(ctx)

        # Add response to history
        self.messages.append(AgentMessage(role="assistant", content=response_text, timestamp=time.time()))
        self._trim_history()

        return {
            "response": response_text,
            "context": context_text,
            "has_llm": self.has_llm,
            "sources": sources,
            "history_length": len(self.messages),
        }

    def summarize(self) -> Dict[str, Any]:
        """Generate a summary of recent activity."""
        ctx = self.collector.collect(self.state_provider)
        context_text = self.collector.format_for_prompt(ctx)

        if self.openai_client:
            prompt = f"Please provide a brief natural-language summary of recent activity in this scene.\n\n{context_text}"
            resp, _ = self._query_llm(prompt, context_text)
            summary = resp
        else:
            summary = self.fallback.summarize(ctx)

        return {
            "summary": summary,
            "context": context_text,
            "has_llm": self.has_llm,
            "object_count": ctx["object_count"],
            "event_count": len(ctx["recent_events"]),
        }

    def suggest(self) -> Dict[str, Any]:
        """Get action recommendations."""
        ctx = self.collector.collect(self.state_provider)
        if self.openai_client:
            context_text = self.collector.format_for_prompt(ctx)
            prompt = f"Based on the current scene, what actions would you recommend the user take?\n\n{context_text}"
            resp, _ = self._query_llm(prompt, context_text)
            return {"recommendations": resp, "has_llm": True}
        else:
            rec = self.fallback._answer_recommend(ctx)
            return {"recommendations": rec, "has_llm": False}

    def _query_llm(self, question: str, context: str) -> Tuple[str, List[str]]:
        """Query OpenAI API."""
        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
            ]
            # Add recent history (last 6)
            for msg in self.messages[-6:]:
                messages.append(msg.to_dict())

            # Add current context
            messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                max_tokens=500,
                temperature=0.7,
                timeout=15.0,
            )
            text = response.choices[0].message.content.strip()
            return text, ["openai"]
        except Exception as e:
            logger.warning(f"OpenAI query failed: {e}, falling back")
            return self.fallback.answer(question, self.collector.collect(self.state_provider)), []

    def _extract_sources(self, ctx: Dict[str, Any]) -> List[str]:
        sources = []
        if ctx["objects"]:
            sources.append(f"objects ({ctx['object_count']})")
        if ctx.get("human_present"):
            sources.append("human_pose")
        if ctx.get("recent_events"):
            sources.append(f"events ({len(ctx['recent_events'])})")
        if ctx.get("intel_interactions"):
            sources.append("scene_graph")
        if ctx.get("physics_bodies", 0) > 0:
            sources.append("physics")
        return sources

    def _trim_history(self) -> None:
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]

    def get_status(self) -> Dict[str, Any]:
        ctx = self.collector.collect(self.state_provider)
        return {
            "has_llm": self.has_llm,
            "model": self.openai_model if self.has_llm else "smart_fallback",
            "history_length": len(self.messages),
            "object_count": ctx["object_count"],
            "human_present": ctx.get("human_present", False),
            "events_count": len(ctx["recent_events"]),
            "camera_running": ctx.get("camera_running", False),
        }

    def reset_conversation(self) -> None:
        self.messages.clear()
