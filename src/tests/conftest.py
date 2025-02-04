import pytest
import numpy as np
import torch
import open3d as o3d
from pathlib import Path

@pytest.fixture
def sample_sehi_data():
    """Generate sample SEHI data for testing."""
    return np.random.rand(100, 100).astype(np.float32)

@pytest.fixture
def sample_point_cloud():
    """Generate sample point cloud for testing."""
    pcd = o3d.geometry.PointCloud()
    points = np.random.rand(1000, 3)
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

@pytest.fixture
def sample_cnn_data():
    """Generate sample data for CNN testing."""
    X = torch.randn(16, 1, 64, 64)  # 16 samples, 1 channel, 64x64 pixels
    y = torch.randint(0, 10, (16,))  # 16 labels, 10 classes
    return X, y 