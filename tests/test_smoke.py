from fastapi.testclient import TestClient

from backend.api.server import app, axis_state


client = TestClient(app)


def setup_function():
    axis_state.reset()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_scene_update_and_search():
    update = client.post(
        "/scene/update",
        json={
            "frame_id": 1,
            "camera_active": True,
            "detections": [
                {
                    "label": "bottle",
                    "score": 0.93,
                    "bbox": {"x": 120, "y": 140, "width": 80, "height": 160},
                },
                {
                    "label": "person",
                    "score": 0.88,
                    "bbox": {"x": 320, "y": 80, "width": 170, "height": 300},
                },
            ],
        },
    )
    assert update.status_code == 200
    state = update.json()
    assert state["semantic_objects"] == 2

    search = client.post("/search", json={"query": "where is the bottle", "top_k": 5})
    assert search.status_code == 200
    payload = search.json()
    assert payload["total"] >= 1
    assert payload["results"][0]["class_name"] == "bottle"


def test_query_returns_scene_answer():
    client.post(
        "/scene/update",
        json={
            "frame_id": 4,
            "camera_active": True,
            "detections": [
                {
                    "label": "chair",
                    "score": 0.9,
                    "bbox": {"x": 250, "y": 180, "width": 180, "height": 200},
                }
            ],
        },
    )

    response = client.post("/query", json={"question": "What do you see?", "context": True})
    assert response.status_code == 200
    assert "chair" in response.json()["response"].lower()
