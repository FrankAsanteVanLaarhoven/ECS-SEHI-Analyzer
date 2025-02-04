import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from scipy.spatial import transform
import open3d as o3d
import logging

@dataclass
class PhotogrammetryParameters:
    """Parameters for photogrammetry processing."""
    min_features: int = 1000
    feature_quality: float = 0.01
    matching_distance: float = 0.7
    ransac_threshold: float = 4.0
    min_inliers: int = 15
    voxel_size: float = 0.05

class PhotogrammetryProcessor:
    """Advanced photogrammetry processing for carbon material analysis."""
    
    def __init__(self, params: Optional[PhotogrammetryParameters] = None):
        self.params = params or PhotogrammetryParameters()
        self.feature_detector = cv2.SIFT_create()
        self.matcher = cv2.BFMatcher_create(cv2.NORM_L2, crossCheck=False)
        
    def process_image_sequence(self, 
                             images: List[np.ndarray],
                             masks: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        """Process image sequence to create 3D reconstruction."""
        try:
            # Extract features from all images
            features = self._extract_features(images, masks)
            
            # Match features between consecutive images
            matches = self._match_features(features)
            
            # Estimate camera poses
            poses = self._estimate_camera_poses(features, matches)
            
            # Generate sparse point cloud
            sparse_cloud = self._generate_sparse_cloud(features, matches, poses)
            
            # Dense reconstruction
            dense_cloud = self._generate_dense_cloud(images, poses, sparse_cloud)
            
            # Surface reconstruction
            mesh = self._reconstruct_surface(dense_cloud)
            
            return {
                'sparse_cloud': sparse_cloud,
                'dense_cloud': dense_cloud,
                'mesh': mesh,
                'camera_poses': poses,
                'features': features
            }
            
        except Exception as e:
            logging.error(f"Error in photogrammetry processing: {str(e)}")
            return self._generate_default_results()

    def _extract_features(self, 
                         images: List[np.ndarray],
                         masks: Optional[List[np.ndarray]] = None) -> List[Dict]:
        """Extract SIFT features from images."""
        features = []
        for i, image in enumerate(images):
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply mask if provided
            if masks and i < len(masks):
                gray = cv2.bitwise_and(gray, gray, mask=masks[i])
            
            # Detect keypoints and compute descriptors
            keypoints, descriptors = self.feature_detector.detectAndCompute(gray, None)
            
            features.append({
                'keypoints': keypoints,
                'descriptors': descriptors,
                'image_size': gray.shape
            })
            
        return features

    def _match_features(self, features: List[Dict]) -> List[List[cv2.DMatch]]:
        """Match features between consecutive image pairs."""
        matches = []
        for i in range(len(features) - 1):
            # Match descriptors
            raw_matches = self.matcher.knnMatch(
                features[i]['descriptors'],
                features[i + 1]['descriptors'],
                k=2
            )
            
            # Apply ratio test
            good_matches = []
            for m, n in raw_matches:
                if m.distance < self.params.matching_distance * n.distance:
                    good_matches.append(m)
            
            matches.append(good_matches)
            
        return matches

    def _estimate_camera_poses(self, 
                             features: List[Dict],
                             matches: List[List[cv2.DMatch]]) -> np.ndarray:
        """Estimate camera poses from feature matches."""
        poses = [np.eye(4)]  # First camera is reference frame
        
        for i, match_set in enumerate(matches):
            if len(match_set) < self.params.min_inliers:
                poses.append(poses[-1])  # Use previous pose if not enough matches
                continue
                
            # Get matched points
            pts1 = np.float32([features[i]['keypoints'][m.queryIdx].pt for m in match_set])
            pts2 = np.float32([features[i+1]['keypoints'][m.trainIdx].pt for m in match_set])
            
            # Essential matrix estimation
            E, mask = cv2.findEssentialMat(
                pts1, pts2,
                focal=1.0, pp=(0., 0.),
                method=cv2.RANSAC,
                prob=0.999,
                threshold=self.params.ransac_threshold
            )
            
            # Recover pose
            _, R, t, _ = cv2.recoverPose(E, pts1, pts2, mask=mask)
            
            # Create transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.ravel()
            
            poses.append(poses[-1] @ T)
        
        return np.array(poses)

    def _generate_sparse_cloud(self,
                             features: List[Dict],
                             matches: List[List[cv2.DMatch]],
                             poses: np.ndarray) -> np.ndarray:
        """Generate sparse point cloud from matches."""
        points = []
        colors = []
        
        for i, match_set in enumerate(matches):
            pts1 = np.float32([features[i]['keypoints'][m.queryIdx].pt for m in match_set])
            pts2 = np.float32([features[i+1]['keypoints'][m.trainIdx].pt for m in match_set])
            
            # Triangulate points
            P1 = poses[i][:3]
            P2 = poses[i+1][:3]
            
            pts4D = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            pts3D = pts4D[:3] / pts4D[3]
            
            points.extend(pts3D.T)
            
        return np.array(points)

    def _generate_dense_cloud(self,
                            images: List[np.ndarray],
                            poses: np.ndarray,
                            sparse_cloud: np.ndarray) -> np.ndarray:
        """Generate dense point cloud using MVS."""
        # Convert data to Open3D format
        cameras = []
        for pose in poses:
            cam = o3d.camera.PinholeCameraParameters()
            cam.extrinsic = pose
            cam.intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=images[0].shape[1],
                height=images[0].shape[0],
                fx=1.0, fy=1.0,
                cx=images[0].shape[1]/2,
                cy=images[0].shape[0]/2
            )
            cameras.append(cam)
        
        # Create dense point cloud
        volume = o3d.pipelines.dense_slam.ScalableTSDFVolume(
            voxel_length=self.params.voxel_size,
            sdf_trunc=0.04,
            color_type=o3d.pipelines.dense_slam.TSDFVolumeColorType.RGB8
        )
        
        for i, image in enumerate(images):
            rgb = o3d.geometry.Image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            depth = o3d.geometry.Image(np.zeros_like(image[:,:,0], dtype=np.float32))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb, depth,
                depth_scale=1000.0,
                depth_trunc=3.0,
                convert_rgb_to_intensity=False
            )
            
            volume.integrate(rgbd, cameras[i].intrinsic, cameras[i].extrinsic)
        
        pcd = volume.extract_point_cloud()
        return np.asarray(pcd.points)

    def _reconstruct_surface(self, point_cloud: np.ndarray) -> o3d.geometry.TriangleMesh:
        """Reconstruct surface mesh from point cloud."""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        
        # Estimate normals
        pcd.estimate_normals()
        
        # Poisson surface reconstruction
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=8,
            width=0,
            scale=1.1,
            linear_fit=False
        )
        
        return mesh

    def _generate_default_results(self) -> Dict[str, Any]:
        """Generate default results for error cases."""
        return {
            'sparse_cloud': np.zeros((100, 3)),
            'dense_cloud': np.zeros((1000, 3)),
            'mesh': o3d.geometry.TriangleMesh(),
            'camera_poses': np.array([np.eye(4)]),
            'features': []
        }

    def integrate_with_sehi(self, 
                          photogrammetry_data: Dict[str, Any],
                          sehi_data: np.ndarray) -> Dict[str, Any]:
        """Integrate photogrammetry reconstruction with SEHI data."""
        try:
            # Align point clouds
            aligned_cloud = self._align_clouds(
                photogrammetry_data['dense_cloud'],
                sehi_data
            )
            
            # Map SEHI data onto 3D reconstruction
            textured_mesh = self._map_sehi_to_mesh(
                photogrammetry_data['mesh'],
                sehi_data
            )
            
            # Generate integrated visualization
            visualization = self._create_integrated_view(
                aligned_cloud,
                textured_mesh,
                sehi_data
            )
            
            return {
                'aligned_cloud': aligned_cloud,
                'textured_mesh': textured_mesh,
                'visualization': visualization
            }
            
        except Exception as e:
            logging.error(f"Error in SEHI integration: {str(e)}")
            return None

    def _align_clouds(self, 
                     photo_cloud: np.ndarray,
                     sehi_cloud: np.ndarray) -> np.ndarray:
        """Align photogrammetry and SEHI point clouds."""
        # Convert to Open3D format
        photo_pcd = o3d.geometry.PointCloud()
        photo_pcd.points = o3d.utility.Vector3dVector(photo_cloud)
        
        sehi_pcd = o3d.geometry.PointCloud()
        sehi_pcd.points = o3d.utility.Vector3dVector(sehi_cloud)
        
        # Initial alignment using FPFH features
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            photo_pcd, sehi_pcd,
            o3d.pipelines.registration.compute_fpfh_feature(photo_pcd),
            o3d.pipelines.registration.compute_fpfh_feature(sehi_pcd),
            max_correspondence_distance=self.params.ransac_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    self.params.ransac_threshold)
            ]
        )
        
        # Fine alignment using ICP
        result = o3d.pipelines.registration.registration_icp(
            photo_pcd, sehi_pcd, self.params.ransac_threshold, result.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        return np.asarray(photo_pcd.transform(result.transformation).points) 