# AXIS - Adaptive eXtended Intelligence System

AXIS is a research-grade embodied AI platform that transforms a standard webcam into a real-time world modeling system capable of understanding 3D environments, building semantic scene representations, tracking humans and objects, predicting future events, and enabling natural language reasoning over the real world.

## Quick Start with Docker

```bash
# Build and run with Docker Compose
docker-compose up -d

# Open the application
open http://localhost:8000
```

## Docker Commands

```bash
# Build image
make build

# Run container
make start

# View logs
make logs

# Stop container
make stop

# Shell into container
make shell
```

## Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Start backend
PYTHONPATH=/workspace/project/AXIS python -m uvicorn backend.api.server:app --host 0.0.0.0 --port 8000

# Open http://localhost:8000
```

## Project Structure

```
AXIS/
├── backend/           # Python backend
│   ├── api/          # FastAPI endpoints
│   ├── core/         # Core utilities (camera, types)
│   ├── scene/        # Scene representation (semantic map, graph)
│   └── ...
├── frontend/         # Web dashboard (Three.js)
├── config/           # Configuration
├── tests/            # Unit tests
├── Dockerfile        # Docker configuration
├── docker-compose.yml # Docker Compose
├── Makefile          # Docker commands
└── requirements.txt  # Python dependencies
```

## Features

- **Webcam ingestion** with browser-based TensorFlow.js COCO-SSD
- **Real-time object detection** and tracking
- **Semantic search** across detected objects
- **3D scene viewer** with Three.js
- **Scene graph** with spatial relationships
- **Natural language Q&A** about the scene
- **WebSocket** real-time sync

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend UI |
| `/health` | GET | Health check |
| `/state` | GET | Current scene state |
| `/scene/update` | POST | Update with detections |
| `/search` | POST | Semantic object search |
| `/query` | POST | Ask questions |
| `/ws/scene` | WS | WebSocket for real-time updates |

## License

Research use only.