# AXIS - Adaptive eXtended Intelligence System

AXIS is a research-grade embodied AI platform that transforms a standard webcam into a real-time world modeling system capable of understanding 3D environments, building semantic scene representations, tracking humans and objects, predicting future events, and enabling natural language reasoning over the real world.

## Quick Start

### Prerequisites

- Python 3.9+
- Modern browser (Chrome/Firefox/Safari)
- Webcam

### 1. Install Dependencies

```bash
cd AXIS
pip install -r requirements.txt
```

If you only want the core functionality (no PyTorch/CUDA models), install just the essentials:

```bash
pip install fastapi uvicorn pydantic pydantic-settings loguru opencv-python-headless numpy
```

### 2. Start the Backend Server

```bash
cd AXIS
PYTHONPATH=$(pwd) python -m uvicorn backend.api.server:app --host 0.0.0.0 --port 8000
```

### 3. Open the Application

Open **http://localhost:8000** in your browser.

### 4. Use It

1. Click **"Start Camera"** — your browser will ask for camera permission
2. Point your camera at objects — the system detects them using TensorFlow.js COCO-SSD (runs in-browser)
3. Detections are sent to the backend, which builds a 3D semantic map and scene graph
4. Watch the **3D Scene View** populate with detected objects
5. Use the **Search** panel to find objects (e.g., "bottle", "chair", "person")
6. Use **"Ask about the scene"** to query what the system sees

### With Docker

```bash
docker-compose up -d
open http://localhost:8005
```

## How It Works

```
Browser (TF.js COCO-SSD)  →  POST /scene/update  →  Backend (FastAPI)
     │                                                    │
     │  Detects objects in-browser                        │  Builds 3D semantic map
     │  Sends labels + bounding boxes                     │  Creates scene graph
     │  Renders 3D scene (Three.js)                       │  Handles search/query
     │                                                    │
     └────────── WebSocket /ws/scene ←────────────────────┘
```

## Project Structure

```
AXIS/
├── backend/           # Python backend
│   ├── api/          # FastAPI endpoints (server.py, routes.py)
│   ├── core/         # Core utilities (camera, types, frame)
│   ├── scene/        # Scene representation (semantic map, scene graph, search)
│   ├── cv/           # Computer vision (segmentation, features)
│   ├── slam/         # SLAM module (droid_slam, map)
│   ├── human/        # Human tracking (future)
│   ├── llm/          # LLM reasoning (future)
│   ├── physics/      # Physics simulation (future)
│   ├── prediction/   # Future prediction (future)
│   └── graph/        # Neo4j integration (future)
├── frontend/         # Web dashboard (Single HTML file with Three.js)
├── config/           # Configuration (settings.py)
├── tests/            # Unit tests
├── Dockerfile        # Docker configuration
├── docker-compose.yml # Docker Compose
├── Makefile          # Docker commands
└── requirements.txt  # Python dependencies
```

## Features (Phase 1 - Spatial Intelligence MVP)

- **Webcam ingestion** — browser-based TensorFlow.js COCO-SSD
- **Real-time object detection** — detects 90 COCO classes in-browser
- **3D scene mapping** — builds a semantic 3D map of detected objects
- **Scene graph** — spatial relationships (near, left_of, right_of, above, below)
- **Semantic search** — find objects by name (e.g., "where is my bottle?")
- **Natural language Q&A** — ask "What do you see?", "How many chairs?", "Where is the person?"
- **WebSocket** — real-time state sync between frontend and backend

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend UI |
| `/health` | GET | Health check |
| `/state` | GET | Current scene state |
| `/scene/update` | POST | Update scene with detections |
| `/scene/reset` | POST | Reset scene |
| `/camera/start` | POST | Enable camera on backend |
| `/camera/stop` | POST | Disable camera on backend |
| `/search` | POST | Semantic object search |
| `/query` | POST | Ask questions about the scene |
| `/trajectory` | GET | Camera trajectory |
| `/map/points` | GET | Point cloud data |
| `/semantic/objects` | GET | All semantic objects |
| `/scene/graph` | GET | Scene graph |
| `/ws/scene` | WS | WebSocket for real-time updates |

## Tests

```bash
PYTHONPATH=$(pwd) python -m pytest tests/ -v
```

## License

Research use only.

## Roadmap

- **Phase 1** ✅ — Webcam ingestion, object detection, 3D mapping, semantic search
- **Phase 2** — Neural scene representation (Gaussian Splatting)
- **Phase 3** — Human digital twin (pose, body, face tracking)
- **Phase 4** — Future world-state prediction
- **Phase 5** — Physics-aware simulation
- **Phase 6** — Scene graph intelligence
- **Phase 7** — LLM-based spatial reasoning