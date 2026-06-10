"""Unit tests for scene graph."""
from backend.scene.scene_graph import SceneGraph, SceneNode, RelationType
from backend.core.types import Vector3


def test_scene_graph_empty():
    sg = SceneGraph()
    assert sg.node_count == 0
    assert sg.edge_count == 0


def test_scene_graph_add_node():
    sg = SceneGraph()
    node = sg.add_node("test_1", "bottle", "object", position=Vector3(1, 2, 3))
    assert node.node_id == "test_1"
    assert node.label == "bottle"
    assert sg.node_count == 1


def test_scene_graph_add_edge():
    sg = SceneGraph()
    sg.add_node("a", "bottle")
    sg.add_node("b", "cup")
    edge = sg.add_edge("a", "b", RelationType.NEAR, confidence=0.8)
    assert edge is not None
    assert edge.relation_type == RelationType.NEAR
    assert edge.confidence == 0.8
    assert sg.edge_count == 1


def test_scene_graph_missing_node():
    sg = SceneGraph()
    sg.add_node("a", "bottle")
    edge = sg.add_edge("a", "missing", RelationType.NEAR)
    assert edge is None


def test_scene_graph_find_path():
    sg = SceneGraph()
    sg.add_node("a", "a")
    sg.add_node("b", "b")
    sg.add_node("c", "c")
    sg.add_edge("a", "b", RelationType.NEAR)
    sg.add_edge("b", "c", RelationType.NEAR)
    path = sg.find_path("a", "c")
    assert path == ["a", "b", "c"]


def test_scene_graph_traverse():
    sg = SceneGraph()
    sg.add_node("root", "root")
    sg.add_node("child1", "child1")
    sg.add_node("child2", "child2")
    sg.add_edge("root", "child1", RelationType.NEAR)
    sg.add_edge("root", "child2", RelationType.NEAR)
    result = sg.traverse("root", max_depth=1)
    assert len(result["children"]) == 2


def test_scene_graph_get_subgraph():
    sg = SceneGraph()
    sg.add_node("center", "center")
    sg.add_node("nearby", "nearby")
    sg.add_node("far", "far")
    sg.add_edge("center", "nearby", RelationType.NEAR)
    sg.add_edge("nearby", "far", RelationType.NEAR)
    sub = sg.get_subgraph("center", radius=1)
    assert "center" in sub.nodes
    assert "nearby" in sub.nodes
    assert "far" not in sub.nodes


def test_scene_graph_remove_node():
    sg = SceneGraph()
    sg.add_node("a", "a")
    sg.add_node("b", "b")
    sg.add_edge("a", "b", RelationType.NEAR)
    assert sg.node_count == 2
    assert sg.edge_count == 1
    sg.remove_node("a")
    assert sg.node_count == 1
    assert sg.edge_count == 0


def test_scene_graph_to_dict():
    sg = SceneGraph()
    sg.add_node("a", "bottle", position=Vector3(1, 2, 3))
    sg.add_node("b", "cup")
    sg.add_edge("a", "b", RelationType.NEAR)
    d = sg.to_dict()
    assert "nodes" in d
    assert "edges" in d
    assert len(d["nodes"]) == 2
    assert len(d["edges"]) == 1


def test_scene_node_timestamps_differ():
    import time
    n1 = SceneNode("a", "test", "object")
    time.sleep(0.001)
    n2 = SceneNode("b", "test", "object")
    assert n1.created_at != n2.created_at, "Timestamps should be instance-specific"
