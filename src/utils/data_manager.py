import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
import h5py
import json
import os
from datetime import datetime
import cv2
import logging
import streamlit as st
from utils.surface_analysis import SurfaceAnalyzer
from utils.defect_analysis import DefectAnalyzer
from utils.chemical_analysis import ChemicalAnalyzer

class DataManager:
    """Manages data import/export for different analysis modes."""
    
    SUPPORTED_FORMATS = {
        'SEHI': ['.h5', '.nxs', '.dat'],
        'ECS': ['.csv', '.txt', '.xlsx'],
        'LiDAR': ['.ply', '.xyz', '.las'],
        'Photogrammetry': ['.jpg', '.png', '.jpeg', '.tiff'],
        'General': ['.json', '.npz']
    }
    
    def __init__(self, base_path: str = "data"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)
        
    def import_data(self, file, format_type: str) -> Dict[str, Any]:
        """Import data from various file formats."""
        if format_type == 'SEHI':
            return self._import_sehi(file)
        elif format_type == 'ECS':
            return self._import_ecs(file)
        elif format_type == 'LiDAR':
            return self._import_lidar(file)
        elif format_type == 'Photogrammetry':
            return self._import_photogrammetry(file)
        else:
            return self._import_general(file)
    
    def export_results(self, results: Dict[str, Any], 
                      format_type: str,
                      filename: Optional[str] = None) -> str:
        """Export analysis results to specified format."""
        if filename is None:
            filename = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
        if format_type == 'report':
            return self._export_report(results, filename)
        elif format_type == 'data':
            return self._export_data(results, filename)
        else:
            return self._export_general(results, filename)
    
    def _import_sehi(self, file) -> Dict[str, Any]:
        """Import SEHI data from various formats."""
        ext = os.path.splitext(file.name)[1].lower()
        
        if ext == '.h5':
            with h5py.File(file, 'r') as f:
                data = {
                    'spectrum': np.array(f['spectrum']),
                    'metadata': dict(f['metadata'].attrs)
                }
        elif ext == '.nxs':
            # Add NeXus format support
            pass
        elif ext == '.dat':
            # Add custom DAT format support
            pass
            
        return data
    
    def _import_ecs(self, file) -> Dict[str, Any]:
        """Import ECS data from various formats."""
        ext = os.path.splitext(file.name)[1].lower()
        
        if ext == '.csv':
            data = pd.read_csv(file)
        elif ext == '.xlsx':
            data = pd.read_excel(file)
        elif ext == '.txt':
            data = pd.read_csv(file, delimiter='\t')
            
        return {'data': data}
    
    def _import_lidar(self, file) -> Dict[str, Any]:
        """Import LiDAR point cloud data."""
        ext = os.path.splitext(file.name)[1].lower()
        
        if ext == '.ply':
            # Add PLY format support
            pass
        elif ext == '.xyz':
            # Add XYZ format support
            pass
        elif ext == '.las':
            # Add LAS format support
            pass
            
        return {'point_cloud': None}  # Replace with actual data
    
    def _import_photogrammetry(self, file) -> Dict[str, Any]:
        """Import photogrammetry image data."""
        try:
            # Read image file
            image_array = cv2.imdecode(
                np.frombuffer(file.read(), np.uint8),
                cv2.IMREAD_COLOR
            )
            
            return {
                'image': image_array,
                'filename': file.name,
                'shape': image_array.shape,
                'type': 'photogrammetry'
            }
        except Exception as e:
            logging.error(f"Error importing photogrammetry data: {str(e)}")
            return None
    
    def _export_report(self, results: Dict[str, Any], filename: str) -> str:
        """Export analysis results as a formatted report."""
        report_path = os.path.join(self.base_path, f"{filename}.pdf")
        # Add report generation logic
        return report_path
    
    def _export_data(self, results: Dict[str, Any], filename: str) -> str:
        """Export raw analysis data."""
        data_path = os.path.join(self.base_path, f"{filename}.npz")
        np.savez_compressed(data_path, **results)
        return data_path 

class DashboardDataManager:
    """Manage data collection from different dashboard components."""
    
    @staticmethod
    def get_sustainability_metrics():
        """Get sustainability metrics."""
        return {
            "energy_efficiency": st.session_state.get('energy_efficiency', 0),
            "resource_usage": st.session_state.get('resource_usage', {}),
            "environmental_impact": st.session_state.get('environmental_impact', {}),
            "recommendations": st.session_state.get('recommendations', "Sustainability recommendations will be added here.")
        } 