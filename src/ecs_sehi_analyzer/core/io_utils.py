import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional

class DataIO:
    """Data import/export utilities"""
    
    @staticmethod
    def _check_dependencies() -> Dict[str, bool]:
        """Check which optional dependencies are available"""
        dependencies = {
            'h5py': False,
            'netcdf4': False,
            'openpyxl': False
        }
        
        try:
            import h5py
            dependencies['h5py'] = True
        except ImportError:
            pass
            
        try:
            import netCDF4
            dependencies['netcdf4'] = True
        except ImportError:
            pass
            
        try:
            import openpyxl
            dependencies['openpyxl'] = True
        except ImportError:
            pass
            
        return dependencies
    
    @staticmethod
    def import_data(file_path: Union[str, Path], format: str) -> Dict[str, Any]:
        """Import data from various formats"""
        available_deps = DataIO._check_dependencies()
        
        if format.lower() == "csv":
            return pd.read_csv(file_path)
        
        elif format.lower() == "excel":
            if not available_deps['openpyxl']:
                raise ImportError("Please install openpyxl to read Excel files: conda install openpyxl")
            return pd.read_excel(file_path)
        
        elif format.lower() == "hdf5":
            if not available_deps['h5py']:
                raise ImportError("Please install h5py to read HDF5 files: conda install h5py")
            import h5py
            with h5py.File(file_path, 'r') as f:
                return {key: f[key][()] for key in f.keys()}
        
        elif format.lower() == "netcdf":
            if not available_deps['netcdf4']:
                raise ImportError("Please install netCDF4 to read NetCDF files: conda install netcdf4")
            import netCDF4 as nc
            with nc.Dataset(file_path, 'r') as f:
                return {var: f.variables[var][:] for var in f.variables}
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def export_data(data: Dict[str, Any], file_path: Union[str, Path], format: str):
        """Export data to various formats"""
        available_deps = DataIO._check_dependencies()
        
        if format.lower() == "csv":
            pd.DataFrame(data).to_csv(file_path)
        
        elif format.lower() == "excel":
            if not available_deps['openpyxl']:
                raise ImportError("Please install openpyxl to write Excel files: conda install openpyxl")
            pd.DataFrame(data).to_excel(file_path)
        
        elif format.lower() == "hdf5":
            if not available_deps['h5py']:
                raise ImportError("Please install h5py to write HDF5 files: conda install h5py")
            import h5py
            with h5py.File(file_path, 'w') as f:
                for key, value in data.items():
                    f.create_dataset(key, data=value)
        
        elif format.lower() == "netcdf":
            if not available_deps['netcdf4']:
                raise ImportError("Please install netCDF4 to write NetCDF files: conda install netcdf4")
            import netCDF4 as nc
            with nc.Dataset(file_path, 'w') as f:
                for key, value in data.items():
                    f.createDimension(key, value.shape[0])
                    f.createVariable(key, value.dtype, (key,))[:] = value
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @staticmethod
    def get_supported_formats() -> Dict[str, bool]:
        """Get list of supported formats based on available dependencies"""
        deps = DataIO._check_dependencies()
        return {
            'CSV': True,  # Always supported
            'Excel': deps['openpyxl'],
            'HDF5': deps['h5py'],
            'NetCDF': deps['netcdf4']
        } 