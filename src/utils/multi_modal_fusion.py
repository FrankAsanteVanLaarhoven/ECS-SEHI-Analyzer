import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA

class MultiModalFusion:
    def __init__(self):
        self.pca = PCA(n_components=3)

    def align_data(self, lidar_data, sehi_data):
        # Align LiDAR and SEHI data using PCA
        combined_data = np.hstack((lidar_data, sehi_data))
        aligned_data = self.pca.fit_transform(combined_data)
        return aligned_data

    def visualize_3d(self, data):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(data)
        o3d.visualization.draw_geometries([point_cloud])