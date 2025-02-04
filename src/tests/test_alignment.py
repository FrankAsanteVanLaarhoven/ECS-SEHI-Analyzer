import pytest
import numpy as np
import open3d as o3d
from app.utils.alignment import PointCloudAligner

class TestPointCloudAligner:
    def test_preprocess_point_cloud(self, sample_point_cloud):
        """Test point cloud preprocessing."""
        aligner = PointCloudAligner()
        processed = aligner.preprocess_point_cloud(sample_point_cloud)
        
        assert isinstance(processed, o3d.geometry.PointCloud)
        assert processed.has_normals()
        
    def test_align_point_clouds(self, sample_point_cloud):
        """Test point cloud alignment."""
        aligner = PointCloudAligner()
        
        # Create slightly transformed copy of point cloud
        target = sample_point_cloud.clone()
        T = np.eye(4)
        T[:3, 3] = [0.1, 0.1, 0.1]  # Add translation
        target.transform(T)
        
        result = aligner.align_point_clouds(sample_point_cloud, target)
        
        assert result is not None
        assert result.fitness_score > 0.0
        assert result.rmse < 0.1 