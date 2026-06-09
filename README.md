# AXIS - Adaptive eXtended Intelligence System

AXIS is a research-grade embodied AI platform that transforms a standard webcam into a real-time world modeling system capable of understanding 3D environments, building semantic scene representations, tracking humans and objects, predicting future events, and enabling natural language reasoning over the real world.

## Project Structure

```
AXIS/
├── backend/           # Python backend
│   ├── api/          # FastAPI endpoints
│   ├── core/         # Core utilities (camera, types)
│   ├── slam/         # Visual SLAM (DROID-SLAM)
│   ├── cv/           # Computer vision (SAM2, CLIP, DINO)
│   ├── scene/        # Scene representation (semantic map, graph)
│   ├── human/        # Human tracking (4D-Humans, SMPL-X)
│   ├── prediction/   # Future prediction
│   ├── physics/      # Physics simulation (MuJoCo)
│   ├── graph/        # Graph database (Neo4j)
│   └── llm/          # LLM reasoning (LangGraph)
├── frontend/         # Web dashboard
├── config/           # Configuration
├── scripts/          # Utility scripts
├── data/             # Data storage
├── models/           # Model checkpoints
├── tests/            # Unit tests
├── requirements.txt  # Python dependencies
└── main.md          # Full specification
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env

# Start backend
python backend/main.py

# Open frontend
open frontend/index.html
```

## API

### REST Endpoints

- `GET /` - Health check
- `GET /state` - Get system state
- `POST /camera/start` - Start camera
- `POST /camera/stop` - Stop camera
- `POST /search` - Semantic object search
- `POST /query` - LLM query
- `GET /trajectory` - Camera trajectory
- `GET /map/points` - 3D map points
- `GET /semantic/objects` - Semantic objects

### WebSocket

Connect to `/ws/{channel}` for real-time updates:
- `camera` - Camera frames
- `scene` - Scene updates
- `human` - Human tracking
- `prediction` - Future predictions

## Phases

1. **Spatial Intelligence MVP** - 3D reconstruction with semantic search
2. **Neural Scene Representation** - Gaussian Splatting
3. **Human Digital Twin** - Full-body avatar
4. **World Model Prediction** - Future state forecasting
5. **Physics Engine** - Interaction simulation
6. **Scene Graph Intelligence** - Symbolic knowledge
7. **LLM Reasoning Layer** - Natural language understanding

## License

Research use only.