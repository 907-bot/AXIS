"""LLM-based reasoning and scene understanding using LangGraph."""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from enum import Enum
import json
from loguru import logger

try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    from langgraph.graph import StateGraph, END
except ImportError:
    ChatOpenAI = None
    StateGraph = None


class ReasoningType(Enum):
    """Types of reasoning tasks."""
    QUERY = "query"
    SUMMARY = "summary"
    PREDICTION = "prediction"
    RECOMMENDATION = "recommendation"
    EXPLANATION = "explanation"


@dataclass
class ReasoningResult:
    """Result from LLM reasoning."""
    reasoning_type: ReasoningType
    query: str
    response: str
    context_used: List[str]
    confidence: float
    sources: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.reasoning_type.value,
            "query": self.query,
            "response": self.response,
            "context_used": self.context_used,
            "confidence": self.confidence,
            "sources": self.sources
        }


@dataclass
class SceneContext:
    """Context information for scene understanding."""
    objects: List[Dict[str, Any]]
    persons: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    recent_events: List[Dict[str, Any]]
    spatial_info: Dict[str, Any]


class SceneAnalyzer:
    """
    Scene understanding using LLM.
    
    Provides:
    - Natural language scene description
    - Object relationship analysis
    - Activity recognition
    - Anomaly detection
    """

    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm

    def describe_scene(self, context: SceneContext) -> str:
        """Generate natural language scene description."""
        if self.llm is None:
            return "Scene description unavailable - no LLM configured"

        # Build scene description prompt
        object_list = ", ".join([
            f"{o.get('class_name', 'object')} at ({o.get('position', {})})"
            for o in context.objects[:10]
        ])

        prompt = f"""Describe the following scene in natural language:

Objects: {object_list}
Persons: {len(context.persons)}
Recent events: {len(context.recent_events)}

Provide a concise, informative description."""

        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Scene description failed: {e}")
            return "Unable to generate description"

    def explain_relationship(
        self,
        subject: str,
        relation: str,
        target: str
    ) -> str:
        """Explain relationship between entities."""
        if self.llm is None:
            return f"{subject} is {relation} {target}"

        prompt = f"""Explain the relationship: {subject} {relation} {target}
Give a brief, clear explanation."""

        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"{subject} is {relation} {target}"


class LLMAgent:
    """
    LLM-powered reasoning agent using LangGraph.
    
    Capabilities:
    - Natural language queries over scene
    - Scene summarization
    - Action recommendations
    - Temporal reasoning
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gpt-4-turbo-preview",
        temperature: float = 0.7
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self.scene_analyzer = None
        self._initialize(api_key)

    def _initialize(self, api_key: Optional[str]):
        """Initialize LLM and components."""
        if ChatOpenAI is None:
            logger.warning("LangChain not installed - LLM features limited")
            return

        try:
            self.llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=api_key,
                request_timeout=30,
            )
            self.scene_analyzer = SceneAnalyzer(self.llm)
            logger.info(f"LLM Agent initialized with {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")

    def query(
        self,
        question: str,
        scene_context: SceneContext,
        reasoning_type: ReasoningType = ReasoningType.QUERY
    ) -> ReasoningResult:
        """
        Process natural language query about scene.
        
        Args:
            question: User question
            scene_context: Current scene information
            reasoning_type: Type of reasoning to perform
            
        Returns:
            Reasoning result with response and confidence
        """
        if self.llm is None:
            return ReasoningResult(
                reasoning_type=reasoning_type,
                query=question,
                response="LLM not available",
                context_used=[],
                confidence=0.0
            )

        # Build context for query
        context_str = self._build_context_string(scene_context)

        # Select prompt based on reasoning type
        prompt = self._build_query_prompt(question, context_str, reasoning_type)

        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            return ReasoningResult(
                reasoning_type=reasoning_type,
                query=question,
                response=response_text,
                context_used=self._extract_context_used(scene_context),
                confidence=0.8  # Placeholder confidence
            )
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return ReasoningResult(
                reasoning_type=reasoning_type,
                query=question,
                response=f"Error: {str(e)}",
                context_used=[],
                confidence=0.0
            )

    def summarize(
        self,
        scene_context: SceneContext,
        time_window: float = 600.0  # Last 10 minutes
    ) -> str:
        """
        Summarize what happened in the scene.
        
        Args:
            scene_context: Scene information
            time_window: Time window in seconds
            
        Returns:
            Natural language summary
        """
        if self.scene_analyzer is None:
            return "Summary unavailable"

        # Get recent events
        recent_events = [
            e for e in scene_context.recent_events
            if scene_context.recent_events  # Filter by time
        ]

        event_summary = ", ".join([
            e.get("type", "event")
            for e in recent_events[-5:]
        ])

        objects = scene_context.objects
        object_summary = f"Scene contains {len(objects)} objects"

        prompt = f"""Summarize the last {time_window/60:.0f} minutes:

Object summary: {object_summary}
Recent events: {event_summary}

Provide a concise narrative of what happened."""

        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return "Unable to generate summary"

    def recommend_actions(
        self,
        scene_context: SceneContext,
        goal: Optional[str] = None
    ) -> List[str]:
        """
        Recommend actions based on scene and goal.
        
        Args:
            scene_context: Current scene
            goal: Optional user goal
            
        Returns:
            List of recommended actions
        """
        if self.llm is None:
            return []

        context_str = self._build_context_string(scene_context)

        prompt = f"""Based on the current scene, recommend actions:

Scene: {context_str}
Goal: {goal or "No specific goal"}

Provide 3-5 actionable recommendations."""

        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            # Parse recommendations (simple extraction)
            recommendations = [
                line.strip("- ").strip()
                for line in response_text.split("\n")
                if line.strip() and line[0].isdigit()
            ]

            return recommendations[:5]
        except Exception as e:
            logger.error(f"Recommendation failed: {e}")
            return []

    def predict_next(
        self,
        scene_context: SceneContext
    ) -> List[str]:
        """
        Predict what will happen next.
        
        Uses scene context and recent patterns.
        """
        if self.llm is None:
            return []

        context_str = self._build_context_string(scene_context)

        prompt = f"""Based on current scene state, predict what will happen next:

Scene: {context_str}

Predict 3 likely next events or actions."""

        try:
            response = self.llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            predictions = [
                line.strip("- ").strip()
                for line in response_text.split("\n")
                if line.strip()
            ]

            return predictions[:3]
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return []

    def explain(self, query: str, scene_context: SceneContext) -> str:
        """
        Explain something about the scene.
        
        Handles "why", "how", "what" questions.
        """
        if self.llm is None:
            return "LLM not available"

        context_str = self._build_context_string(scene_context)

        prompt = f"""Explain the following about the scene:

Question: {query}

Scene Context: {context_str}

Provide a clear, informative explanation."""

        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Explanation failed: {e}")
            return "Unable to provide explanation"

    def _build_context_string(self, context: SceneContext) -> str:
        """Build context string from scene context."""
        parts = []

        # Objects
        if context.objects:
            obj_str = ", ".join([
                f"{o.get('class_name', 'object')}"
                for o in context.objects[:10]
            ])
            parts.append(f"Objects: {obj_str}")

        # Persons
        if context.persons:
            parts.append(f"Persons detected: {len(context.persons)}")

        # Relationships
        if context.relationships:
            rel_str = ", ".join([
                f"{r.get('subject')} {r.get('relation')} {r.get('object')}"
                for r in context.relationships[:5]
            ])
            parts.append(f"Relationships: {rel_str}")

        return "\n".join(parts)

    def _build_query_prompt(
        self,
        question: str,
        context: str,
        reasoning_type: ReasoningType
    ) -> List:
        """Build prompt for query."""
        system_prompt = """You are AXIS, an embodied AI assistant with full understanding of the 3D scene.
You have access to:
- Object positions and classifications
- Person locations and activities
- Spatial relationships between entities
- Recent events and temporal patterns

Provide accurate, concise responses based on the scene context."""

        user_prompt = f"""Context:
{context}

Question: {question}

Answer:"""

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

    def _extract_context_used(self, context: SceneContext) -> List[str]:
        """Extract which context elements were used."""
        used = []
        if context.objects:
            used.append(f"{len(context.objects)} objects")
        if context.persons:
            used.append(f"{len(context.persons)} persons")
        if context.relationships:
            used.append(f"{len(context.relationships)} relationships")
        return used


class ReasoningChain:
    """Build reasoning chains for complex queries."""

    def __init__(self, llm_agent: LLMAgent):
        self.agent = llm_agent

    def chain_of_thought(
        self,
        question: str,
        scene_context: SceneContext
    ) -> List[Dict[str, str]]:
        """
        Perform chain-of-thought reasoning.
        
        Breaks down complex question into steps.
        """
        steps = []

        # Step 1: Identify what's being asked
        decomposition_prompt = f"""Break down this question into simpler steps:
        
Question: {question}

Steps:"""

        if self.agent.llm:
            try:
                response = self.agent.llm.invoke(decomposition_prompt)
                step_text = response.content if hasattr(response, 'content') else str(response)

                # Parse steps (simplified)
                for line in step_text.split("\n"):
                    if line.strip() and ("step" in line.lower() or line[0].isdigit()):
                        steps.append({"question": line.strip(), "answer": ""})
            except Exception as e:
                logger.error(f"Decomposition failed: {e}")

        # Step 2: Answer each step
        for step in steps:
            result = self.agent.query(
                step["question"],
                scene_context,
                ReasoningType.QUERY
            )
            step["answer"] = result.response

        return steps