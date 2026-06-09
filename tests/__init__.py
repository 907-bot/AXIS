"""Unit tests for AXIS backend."""
import pytest
import numpy as np
from dataclasses import asdict

# Import modules to test
from backend.core.types import Vector3, Quaternion, Pose, Point3D
from backend.core.camera import CameraCapture, CameraFrame
from backend.slam.map import PointCloudMap
from backend.scene.semantic_map import SemanticMap, MapObject
from backend.scene.scene_graph import SceneGraph, RelationType


class TestCoreTypes:
    """Test core data types."""

    def test_vector3_operations(self):
        v1 = Vector3(1, 2, 3)
        v2 = Vector3(4, 5, 6)
        
        # Addition
        v3 = v1 + v2
        assert v3.x == 5
        assert v3.y == 7
        assert v3.z == 9
        
        # Subtraction
        v4 = v2 - v1
        assert v4.x == 3
        assert v4.y == 3
        assert v4.z == 3
        
        # Scalar multiplication
        v5 = v1 * 2
        assert v5.x == 2
        assert v5.y == 4
        assert v5.z == 6
        
        # Magnitude
        mag = Vector3(3, 4, 0).magnitude()
        assert abs(mag - 5.0) < 0.001

    def test_quaternion_rotation(self):
        q = Quaternion(w=1, x=0, y=0, z=0)
        matrix = q.to_rotation_matrix()
        
        # Identity quaternion should give identity matrix
        assert np.allclose(matrix, np.eye(3))

    def test_pose_matrix_conversion(self):
        pose = Pose(
            position=Vector3(1, 2, 3),
            orientation=Quaternion(w=1, x=0, y=0, z=0),
            timestamp=0.0
        )
        
        matrix = pose.to_matrix()
        assert matrix.shape == (4, 4)
        assert np.allclose(matrix[:3, 3], [1, 2, 3])


class TestCameraCapture:
    """Test camera capture functionality."""

    def test_camera_initialization(self):
        camera = CameraCapture(
            device_id=0,
            width=640,
            height=480,
            fps=30
        )
        
        assert camera.device_id == 0
        assert camera.width == 640
        assert camera.height == 480
        assert camera.fps == 30

    def test_intrinsics_computation(self):
        camera = CameraCapture(width=640, height=480)
        K = camera.intrinsics
        
        assert K.shape == (3, 3)
        assert K[0, 0] == K[1, 1]  # fx ≈ fy for square pixels
        assert K[0, 2] == 320  # cx = width / 2
        assert K[1, 2] == 240  # cy = height / 2


class TestSemanticMap:
    """Test semantic map functionality."""

    def test_add_object(self):
        smap = SemanticMap()
        
        obj = smap.add_object(
            object_id="obj1",
            class_name="bottle",
            position=Vector3(1, 0.5, 2),
            size=Vector3(0.1, 0.2, 0.1),
            confidence=0.9
        )
        
        assert obj.object_id == "obj1"
        assert obj.class_name == "bottle"
        assert len(smap.objects) == 1

    def test_search_by_class(self):
        smap = SemanticMap()
        
        smap.add_object("obj1", "bottle", Vector3(0, 0, 0), Vector3(1, 1, 1))
        smap.add_object("obj2", "cup", Vector3(1, 0, 0), Vector3(1, 1, 1))
        smap.add_object("obj3", "bottle", Vector3(2, 0, 0), Vector3(1, 1, 1))
        
        bottles = smap.get_objects_by_class("bottle")
        assert len(bottles) == 2

    def test_spatial_query(self):
        smap = SemanticMap()
        
        smap.add_object("obj1", "test", Vector3(0, 0, 0), Vector3(1, 1, 1))
        smap.add_object("obj2", "test", Vector3(2, 0, 0), Vector3(1, 1, 1))
        
        results = smap.get_objects_in_radius(Vector3(0, 0, 0), 0.5)
        assert len(results) == 1


class TestSceneGraph:
    """Test scene graph functionality."""

    def test_add_nodes_and_edges(self):
        graph = SceneGraph()
        
        graph.add_node("person", "John", "person")
        graph.add_node("object", "Bottle", "object")
        
        graph.add_edge("person", "object", RelationType.HOLDS)
        
        assert graph.node_count == 2
        assert graph.edge_count == 1

    def test_traverse_graph(self):
        graph = SceneGraph()
        
        graph.add_node("a", "A", "type")
        graph.add_node("b", "B", "type")
        graph.add_node("c", "C", "type")
        
        graph.add_edge("a", "b", RelationType.NEAR)
        graph.add_edge("b", "c", RelationType.NEAR)
        
        result = graph.traverse("a", max_depth=2)
        assert result["id"] == "a"
        assert len(result["children"]) > 0

    def test_find_path(self):
        graph = SceneGraph()
        
        graph.add_node("1", "Node1", "type")
        graph.add_node("2", "Node2", "type")
        graph.add_node("3", "Node3", "type")
        
        graph.add_edge("1", "2", RelationType.NEAR)
        graph.add_edge("2", "3", RelationType.NEAR)
        
        path = graph.find_path("1", "3")
        assert path == ["1", "2", "3"]


class TestPointCloudMap:
    """Test point cloud map."""

    def test_add_points(self):
        pcmap = PointCloudMap()
        
        pid = pcmap.add_map_point(Vector3(0, 0, 0))
        assert pid == 0
        
        pid2 = pcmap.add_map_point(Vector3(1, 1, 1))
        assert pid2 == 1
        
        assert pcmap.point_count == 2

    def test_get_point_cloud(self):
        pcmap = PointCloudMap()
        
        pcmap.add_map_point(Vector3(0, 0, 0), color=(255, 0, 0))
        pcmap.add_map_point(Vector3(1, 1, 1), color=(0, 255, 0))
        
        points, colors = pcmap.get_point_cloud()
        
        assert points.shape == (2, 3)
        assert colors.shape == (2, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])