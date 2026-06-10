"""Unit tests for physics engine."""
from backend.physics import PhysicsEngine, sim_force_vectors


def test_physics_engine_initial_state():
    p = PhysicsEngine()
    state = p.get_state()
    assert state["body_count"] == 0
    assert state["time"] == 0.0
    assert state["steps"] == 0


def test_physics_init_from_objects():
    p = PhysicsEngine()
    objects = [
        {"object_id": "bottle_1", "class_name": "bottle", "position": {"x": 1, "y": 0.5, "z": 2}, "size": {"x": 0.1, "y": 0.2, "z": 0.1}},
        {"object_id": "cup_1", "class_name": "cup", "position": {"x": -1, "y": 0.3, "z": 2.5}, "size": {"x": 0.08, "y": 0.12, "z": 0.08}},
    ]
    state = p.init_from_objects(objects)
    assert state["body_count"] == 2
    assert "bottle_1" in [b["object_id"] for b in state["bodies"]]
    assert "cup_1" in [b["object_id"] for b in state["bodies"]]


def test_physics_step_gravity():
    p = PhysicsEngine()
    p.init_from_objects([
        {"object_id": "ball", "class_name": "sports ball", "position": {"x": 0, "y": 2, "z": 0}, "size": {"x": 0.2, "y": 0.2, "z": 0.2}},
    ])
    y0 = p.get_state()["bodies"][0]["position"]["y"]
    p.step(dt=0.016)
    y1 = p.get_state()["bodies"][0]["position"]["y"]
    assert y1 < y0, "Object should fall under gravity"


def test_physics_push():
    p = PhysicsEngine()
    p.init_from_objects([
        {"object_id": "ball", "class_name": "sports ball", "position": {"x": 0, "y": 1, "z": 0}, "size": {"x": 0.2, "y": 0.2, "z": 0.2}},
    ])
    p.push("ball", "forward", 50.0)
    v = p.get_state()["bodies"][0]["velocity"]
    assert abs(v["z"]) > 0, "Push should create forward velocity"


def test_physics_reset():
    p = PhysicsEngine()
    p.init_from_objects([
        {"object_id": "ball", "class_name": "sports ball", "position": {"x": 0, "y": 1, "z": 0}, "size": {"x": 0.2, "y": 0.2, "z": 0.2}},
    ])
    p.reset()
    assert p.get_state()["body_count"] == 0


def test_physics_force_vectors():
    forces = sim_force_vectors()
    assert "gentle_push" in forces
    assert "hard_push" in forces
    assert len(forces["gentle_push"]) == 3
