# utils/alignment.py

import numpy as np
from open3d import pipelines

def align_sehi_lidar(sehi_image, point_cloud, max_distance=0.05):
    """
    Align SEHI image and LiDAR point cloud using ICP.
    
    Args:
        sehi_image (numpy.ndarray): SEHI image.
        point_cloud (open3d.geometry.PointCloud): LiDAR point cloud.
        max_distance (float): Maximum correspondence distance for ICP.
    
    Returns:
        open3d.geometry.PointCloud: Aligned point cloud.
    """
    try:
        # Convert SEHI image to point cloud
        sehi_points = np.argwhere(sehi_image > 0)
        sehi_pcd = o3d.geometry.PointCloud()
        sehi_pcd.points = o3d.utility.Vector3dVector(sehi_points)
        
        # Perform ICP alignment
        icp_result = pipelines.registration.registration_icp(
            sehi_pcd, point_cloud, max_distance
        )
        return icp_result.transformed_point_cloud
    except Exception as e:
        print(f"Error aligning SEHI and LiDAR data: {e}")
        return None