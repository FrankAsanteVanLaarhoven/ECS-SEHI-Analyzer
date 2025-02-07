from pydantic import BaseModel
from typing import List

class Defect(BaseModel):
    coordinates: List[float]
    type: str
    severity: float
    area: float
    confidence: float
