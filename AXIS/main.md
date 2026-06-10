# AXIS — Adaptive eXtended Intelligence System

## Overview

AXIS transforms a standard webcam into a real-time scene understanding system. It runs entirely in the browser for ML inference (no GPU required) and uses a Python backend for state management, scene graph reasoning, physics simulation, and LLM-based querying.

---

## Architecture

| Layer | Technology | What it does |
|-------|-----------|-------------|
| **Frontend** | Single HTML file + Three.js + TensorFlow.js | Webcam capture, object detection (COCO-SSD), pose estimation (MoveNet), 3D rendering |
| **Backend** | Python + FastAPI + Uvicorn | Scene state, search, prediction engine, physics engine, scene graph intelligence, LLM agent |
| **Storage** | In-memory (current), PostgreSQL + Qdrant (composed, planned) | Scene data, embeddings |

### Data Flow

```
Webcam → TensorFlow.js (COCO-SSD + MoveNet) → FastAPI backend
                                                    │
                          ┌─────────────────────────┼─────────────────────────┐
                          ▼                         ▼                         ▼
                   Scene Graph               Prediction Engine          Physics Engine
                  (spatial + semantic)       (trajectory extrapolation)  (rigid body simulation)
                          │                         │                         │
                          └─────────────────────────┼─────────────────────────┘
                                                    ▼
                                            Reasoning Agent
                                        (LLM or SmartFallback)
                                                    │
                                                    ▼
                                            Frontend Dashboard
                                          (Three.js 3D viewer)
```

---

## Phases

### Phase 1 — Spatial Intelligence MVP
- **What's built**: Webcam → COCO-SSD object detection → 3D position projection → semantic search
- **Detection**: Browser-based TensorFlow.js (no server GPU needed)
- **3D Projection**: 2D bounding boxes projected into a simulated 3D frustum using heuristic depth estimation
- **Search**: Hash-based embedding with synonym normalization (deterministic, no external API)
- **Camera**: Simulated sinusoidal trajectory (not real SLAM)

### Phase 2 — Neural Scene Representation
- **What's built**: Colored 3D sphere rendering with custom Three.js shaders
- **Gaussian "splats"**: `SemanticGaussian` dataclass storing position/color/opacity/scale/rotation with covariance matrix; rendered as colored spheres with glow shader
- **Note**: This is a visual approximation of Gaussian Splatting concepts, not actual 3D Gaussian Splatting (which requires CUDA + gsplat)

### Phase 3 — Human Digital Twin
- **What's built**: Real-time skeleton tracking via MoveNet (browser), joint angle computation, balance/stability analytics
- **Rendering**: 3D skeleton with colored joints and bones overlaid on the Three.js scene
- **Analytics**: Joint angles (elbow, knee, shoulder), balance score, motion velocity, stability variance
- **Note**: Uses 17-keypoint MoveNet, not SMPL-X or 4D-Humans (no GPU available)

### Phase 4 — World Model Prediction
- **What's built**: Trajectory extrapolation via linear regression, human action classification from pose history
- **Object prediction**: Position history → linear least-squares fit → future position
- **Action prediction**: Velocity thresholds + torso compression heuristic → stationary/walking/sitting/reaching
- **Note**: Not using video transformers, diffusion models, or latent world models

### Phase 5 — Physics Engine
- **What's built**: Rigid-body simulation with sphere-sphere and sphere-ground collision
- **Integration**: Semi-implicit Euler, linear damping, restitution, friction, ground plane
- **Controls**: Push, drop, rotate, apply force via API
- **Note**: Custom physics (not MuJoCo, not NVIDIA Warp); rotation is a linear impulse approximation

### Phase 6 — Scene Graph Intelligence
- **What's built**: Dynamic spatial-relationship graph built from object positions
- **Relations**: NEAR, LEFT_OF, RIGHT_OF, ABOVE, BELOW (distance-threshold based)
- **Events**: Object detection/removal, interaction synthesis
- **Person tree**: Hierarchical depth-2 traversal of relationships
- **Note**: In-memory scene graph (not Neo4j); no Kafka event streaming

### Phase 7 — LLM Reasoning Layer
- **What's built**: Context collection + OpenAI API (optional) or SmartFallback template engine
- **With API key**: GPT-4o-mini with system prompt, conversation history, full scene context
- **Without API key**: Template-based answering (where, how many, describe, predict, recommend)
- **Note**: Not using LangGraph; fallback is a rule engine, not an LLM

---

## Tech Stack (Actual)

### Backend
- Python 3.11 + FastAPI + Uvicorn
- NumPy, SciPy, OpenCV-Python-Headless
- Loguru for logging
- Pydantic for validation

### Frontend
- Single HTML file (no build step, no npm)
- Three.js (CDN) for 3D rendering
- TensorFlow.js + COCO-SSD + MoveNet (CDN) for browser ML

### Infrastructure
- Docker Compose (3 containers: axis-server, postgres, qdrant)
- Note: PostgreSQL and Qdrant containers are declared but not yet connected to the application

---

## Components vs Reality

| Documented | Actual Implementation |
|-----------|---------------------|
| DROID-SLAM, SAM2, DINOv2, CLIP | Not used. Detection is browser TF.js |
| 3D Gaussian Splatting (gsplat) | Colored spheres with glow shader. Not true gsplat |
| SMPL-X, 4D-Humans, Blender | Not used. MoveNet 17-keypoint skeleton only |
| Video Transformers, Diffusion Models | Not used. Linear regression + velocity thresholds |
| MuJoCo, NVIDIA Warp, PhysGaussian | Not used. Custom Euler integrator with sphere collisions |
| Neo4j, Kafka | Not used. In-memory distance-based graph |
| LangGraph | Not used. Custom agent with optional OpenAI |
| Next.js, React, React Three Fiber | Not used. Single HTML file with Three.js CDN |
| PyTorch3D, CUDA, RTX 5090 | Not used. All ML runs in browser via WebGL |
| Kubernetes | Not used. Single docker-compose deployment |

---

## Success Criteria (What Actually Ships)

1. **Phase 1**: Searchable 3D scene from browser webcam
2. **Phase 2**: Colored 3D spheres with semantic labels in Three.js
3. **Phase 3**: Real-time skeleton overlay with motion analytics
4. **Phase 4**: Trajectory extrapolation and action classification
5. **Phase 5**: Interactive rigid-body physics sandbox
6. **Phase 6**: Dynamic scene graph with spatial relationships
7. **Phase 7**: Natural language querying (LLM or fallback)

---

## Running

```bash
docker compose up -d --build
# Open http://localhost:8005
```

Set `OPENAI_API_KEY` environment variable to enable LLM reasoning (optional).
