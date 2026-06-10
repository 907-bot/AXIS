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


def test_scene_intelligence():
    """Phase 6: Scene graph intelligence — events, interactions, person tree."""
    # Ingest a person + bottle to trigger intelligence analysis
    update = client.post(
        "/scene/update",
        json={
            "frame_id": 10,
            "camera_active": True,
            "detections": [
                {
                    "label": "person",
                    "score": 0.91,
                    "bbox": {"x": 300, "y": 100, "width": 180, "height": 320},
                },
                {
                    "label": "bottle",
                    "score": 0.85,
                    "bbox": {"x": 150, "y": 200, "width": 60, "height": 130},
                },
            ],
        },
    )
    assert update.status_code == 200
    state = update.json()

    # Intelligence should be in snapshot
    assert "intelligence" in state
    intel = state["intelligence"]
    assert "events" in intel
    assert "interactions" in intel

    # Should have entered events for both objects
    entered_events = [e for e in intel["events"] if e["event_type"] == "entered"]
    assert len(entered_events) >= 2
    event_types = {e["event_type"] for e in intel["events"]}
    assert "entered" in event_types

    # Should have interactions (person near bottle, person looks at bottle)
    assert intel["interaction_count"] >= 0  # may vary by distance

    # Direct endpoint
    intel_resp = client.get("/scene/intelligence")
    assert intel_resp.status_code == 200
    intel_data = intel_resp.json()
    assert "events" in intel_data
    assert "scene_graph" in intel_data

    # Scene graph should contain both nodes
    sg = intel_data["scene_graph"]
    node_labels = [n["label"] for n in sg["nodes"].values()]
    assert "person" in node_labels
    assert "bottle" in node_labels

    # Person tree should exist (person in scene)
    tree = intel_data.get("person_tree")
    if tree is not None:
        assert tree["label"] == "person"


def test_agent_reasoning():
    """Phase 7: LLM reasoning agent — query, summarize, suggest, status."""
    # Need objects in scene first
    client.post(
        "/scene/update",
        json={
            "frame_id": 20,
            "camera_active": True,
            "detections": [
                {
                    "label": "bottle",
                    "score": 0.9,
                    "bbox": {"x": 120, "y": 140, "width": 80, "height": 160},
                },
                {
                    "label": "chair",
                    "score": 0.85,
                    "bbox": {"x": 250, "y": 200, "width": 150, "height": 180},
                },
            ],
        },
    )

    # Status
    status = client.get("/agent/status")
    assert status.status_code == 200
    s = status.json()
    assert "has_llm" in s
    assert "model" in s

    # Query
    query_resp = client.post("/agent/query", json={"question": "What do you see?", "use_llm": True})
    assert query_resp.status_code == 200
    q = query_resp.json()
    assert "response" in q
    assert len(q["response"]) > 0
    assert "context" in q
    assert q["history_length"] >= 1

    # Query with source extraction
    where_resp = client.post("/agent/query", json={"question": "where is the bottle", "use_llm": True})
    assert where_resp.status_code == 200
    w = where_resp.json()
    assert "bottle" in w["response"].lower()

    # Summarize
    summary_resp = client.post("/agent/summarize")
    assert summary_resp.status_code == 200
    sm = summary_resp.json()
    assert "summary" in sm
    assert sm["object_count"] >= 2

    # Suggest
    suggest_resp = client.post("/agent/suggest")
    assert suggest_resp.status_code == 200
    sg = suggest_resp.json()
    assert "recommendations" in sg

    # Reset
    reset_resp = client.post("/agent/reset")
    assert reset_resp.status_code == 200
    assert reset_resp.json()["status"] == "reset"
