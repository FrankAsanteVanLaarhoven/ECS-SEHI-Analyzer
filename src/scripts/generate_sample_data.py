import os
import numpy as np
from pathlib import Path
from app.utils.preprocessing import generate_sehi_image, generate_lidar_point_cloud
from app.utils.ecs_data_generator import ECSDataGenerator
import json
import cv2
import open3d as o3d

def main():
    # Create directories if they don't exist
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    
    for subdir in ["sehi_images", "lidar_data", "ecs_materials"]:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Generate sample data
    num_samples = 100
    ecs_generator = ECSDataGenerator()
    
    # Generate and save ECS materials
    materials = ecs_generator.generate_dataset(num_samples)
    for material in materials:
        material_path = data_dir / "ecs_materials" / f"{material.material_id}.json"
        with open(material_path, 'w') as f:
            json.dump(material.to_dict(), f, indent=2)
        
        # Generate corresponding SEHI image
        sehi_image = generate_sehi_image()
        cv2.imwrite(
            str(data_dir / "sehi_images" / f"{material.material_id}.png"),
            sehi_image
        )
        
        # Generate corresponding LiDAR data
        point_cloud = generate_lidar_point_cloud()
        o3d.io.write_point_cloud(
            str(data_dir / "lidar_data" / f"{material.material_id}.pcd"),
            point_cloud
        )

if __name__ == "__main__":
    main() 