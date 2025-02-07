from pydantic import BaseModel, validator
import numpy as np
from typing import Union

class ChemicalMap(BaseModel):
    data: Union[np.ndarray, list]
    resolution: int
    units: str = "eV"
    metadata: dict = {}

    @validator('data')
    def validate_data(cls, value):
        if isinstance(value, list):
            value = np.array(value)
        if not isinstance(value, np.ndarray):
            raise ValueError("Data must be numpy array or convertible to array")
        if value.ndim not in (2, 3):
            raise ValueError("Chemical map must be 2D or 3D array")
        return value

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            np.ndarray: lambda v: v.tolist()
        }
