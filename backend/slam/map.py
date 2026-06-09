"""Point cloud map management for 3D reconstruction."""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
import numpy as np
import open3d as o3d
from loguru import logger

from ..core.types import Vector3, Quaternion, Point3D, Pose


@dataclass
class MapPoint:
    """3D map point with observation history."""
    point_id: int
    position: Vector3
    normal: Optional[Vector3] = None
    color: Optional[Tuple[int, int, int]] = None
    observations: List[Tuple[int, int]] = field(default_factory=list)  # (keyframe_id, keypoint_idx)
    descriptor: Optional[np.ndarray] = None
    depth: float = 0.0
    uncertainty: float = 1.0
    is_valid: bool = True
    last_observed: float = 0.0


@dataclass 
class KeyFrame:
    """Keyframe for mapping."""
    kf_id: int
    pose: Pose
    timestamp: float = 0.0
    connected_kf_ids: Set[int] = field(default_factory=set)
    map_point_ids: Set[int] = field(default_factory=set)


class PointCloudMap:
    """Manages 3D point cloud map and keyframes."""

    def __init__(self, max_points: int = 100000):
        self.max_points = max_points
        self.map_points: Dict[int, MapPoint] = {}
        self.keyframes: Dict[int, KeyFrame] = {}
        self.next_point_id = 0
        self.next_kf_id = 0

        # Statistics
        self.total_observations = 0

    def add_map_point(self, position: Vector3, **kwargs) -> int:
        """Add new map point."""
        point_id = self.next_point_id
        self.next_point_id += 1

        mp = MapPoint(
            point_id=point_id,
            position=position,
            color=kwargs.get("color"),
            normal=kwargs.get("normal"),
            depth=kwargs.get("depth", 0.0),
            uncertainty=kwargs.get("uncertainty", 1.0)
        )

        self.map_points[point_id] = mp
        return point_id

    def add_keyframe(self, pose: Pose) -> int:
        """Add new keyframe."""
        kf_id = self.next_kf_id
        self.next_kf_id += 1

        kf = KeyFrame(kf_id=kf_id, pose=pose)
        self.keyframes[kf_id] = kf
        return kf_id

    def connect_keyframes(self, kf1_id: int, kf2_id: int):
        """Connect two keyframes."""
        if kf1_id in self.keyframes and kf2_id in self.keyframes:
            self.keyframes[kf1_id].connected_kf_ids.add(kf2_id)
            self.keyframes[kf2_id].connected_kf_ids.add(kf1_id)

    def add_observation(self, point_id: int, kf_id: int, keypoint_idx: int):
        """Add observation of map point from keyframe."""
        if point_id in self.map_points:
            self.map_points[point_id].observations.append((kf_id, keypoint_idx))
            self.total_observations += 1

        if kf_id in self.keyframes:
            self.keyframes[kf_id].map_point_ids.add(point_id)

    def cull_invalid_points(self, max_age: float = 10.0, current_time: float = 0.0):
        """Remove old, unreliable map points."""
        to_remove = []
        for pid, mp in self.map_points.items():
            if mp.last_observed > 0 and current_time - mp.last_observed > max_age:
                if len(mp.observations) < 2:
                    to_remove.append(pid)

        for pid in to_remove:
            del self.map_points[pid]

        logger.debug(f"Culled {len(to_remove)} invalid map points")

    def get_point_cloud(self) -> np.ndarray:
        """Get point cloud as Nx3 numpy array."""
        positions = []
        colors = []
        
        for mp in self.map_points.values():
            if mp.is_valid:
                positions.append([mp.position.x, mp.position.y, mp.position.z])
                if mp.color:
                    colors.append([c / 255.0 for c in mp.color])
                else:
                    colors.append([0.5, 0.5, 0.5])

        points = np.array(positions) if positions else np.zeros((0, 3))
        colors = np.array(colors) if colors else np.zeros((0, 3))
        
        return points, colors

    def to_open3d(self) -> o3d.geometry.PointCloud:
        """Convert to Open3D point cloud."""
        points, colors = self.get_point_cloud()
        
        pcd = o3d.geometry.PointCloud()
        if len(points) > 0:
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        
        return pcd

    def densify(self, voxel_size: float = 0.05):
        """Densify point cloud using voxel grid filtering."""
        pcd = self.to_open3d()
        downpcd = pcd.voxel_down_sample(voxel_size)
        
        # Clear and repopulate
        old_count = len(self.map_points)
        self.map_points.clear()
        
        points = np.asarray(downpcd.points)
        colors = np.asarray(downpcd.colors) if downpcd.has_colors() else None
        
        for i, pt in enumerate(points):
            pos = Vector3(x=float(pt[0]), y=float(pt[1]), z=float(pt[2]))
            color = None
            if colors is not None:
                color = tuple(int(c * 255) for c in colors[i])
            
            self.add_map_point(pos, color=color)

        logger.info(f"Densified from {old_count} to {len(self.map_points)} points")

    def segment_plane(self, distance_threshold: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
        """Segment largest plane (floor/wall) from point cloud."""
        pcd = self.to_open3d()
        
        if len(pcd.points) < 100:
            return np.array([]), np.array([])

        try:
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=distance_threshold,
                ransac_n=3,
                num_iterations=1000
            )
            
            inlier_cloud = pcd.select_by_index(inliers)
            outlier_cloud = pcd.select_by_index(inliers, invert=True)
            
            return np.asarray(inlier_cloud.points), np.asarray(outlier_cloud.points)
        except Exception as e:
            logger.error(f"Plane segmentation failed: {e}")
            return np.array([]), np.asarray(pcd.points)

    def save(self, path: str):
        """Save map to file."""
        pcd = self.to_open3d()
        o3d.io.write_point_cloud(path, pcd)
        logger.info(f"Saved point cloud to {path}")

    def load(self, path: str):
        """Load map from file."""
        pcd = o3d.io.read_point_cloud(path)
        
        self.map_points.clear()
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors) if pcd.has_colors() else None
        
        for i, pt in enumerate(points):
            pos = Vector3(x=float(pt[0]), y=float(pt[1]), z=float(pt[2]))
            color = None
            if colors is not None:
                color = tuple(int(c * 255) for c in colors[i])
            
            self.add_map_point(pos, color=color)

        logger.info(f"Loaded {len(self.map_points)} points from {path}")

    @property
    def point_count(self) -> int:
        return len(self.map_points)

    @property
    def keyframe_count(self) -> int:
        return len(self.keyframes)

    def get_stats(self) -> Dict:
        """Get map statistics."""
        depths = [mp.depth for mp in self.map_points.values() if mp.depth > 0]
        return {
            "map_points": len(self.map_points),
            "keyframes": len(self.keyframes),
            "observations": self.total_observations,
            "mean_depth": np.mean(depths) if depths else 0,
            "valid_points": sum(1 for mp in self.map_points.values() if mp.is_valid)
        }