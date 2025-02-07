from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Optional

class CarbonMaterial(BaseModel):
    """Material data model with validation"""
    material_id: str
    graphene_content: float
    surface_area: float
    degradation_rate: float
    
    class Config:
        validate_assignment = True

class ECSAnalyzer:
    def __init__(self):
        self.materials_db = []
        self.current_material = None
        self.current_degradation_rate = None
        self.key_insights = []
    
    def add_material(self, material: CarbonMaterial) -> Dict:
        """Add new carbon material to research database"""
        self.materials_db.append(material)
        self.current_material = material
        return {"status": "success", "material_id": material.material_id}

    def predict_degradation(self, material_id: str) -> Dict:
        """Predict degradation using SEHI data"""
        material = next(m for m in self.materials_db if m.material_id == material_id)
        self.current_degradation_rate = material.degradation_rate
        
        return {
            "predicted_lifetime": (1 - material.degradation_rate) * 1000,
            "stability_index": material.surface_area * 0.8,
            "confidence": 0.92
        }
        
    def analyze_surface_properties(self, material_id: str) -> Dict:
        """Analyze surface properties of material"""
        material = next(m for m in self.materials_db if m.material_id == material_id)
        
        analysis = {
            "surface_area": material.surface_area,
            "graphene_quality": material.graphene_content / 100,
            "defect_density": self._calculate_defect_density(material),
            "recommendations": self._generate_recommendations(material)
        }
        
        self.key_insights = [
            f"Surface area: {analysis['surface_area']:.2f} m²/g",
            f"Graphene quality: {analysis['graphene_quality']:.2%}",
            f"Defect density: {analysis['defect_density']:.2e} defects/cm²"
        ]
        
        return analysis
    
    def _calculate_defect_density(self, material: CarbonMaterial) -> float:
        """Calculate defect density based on material properties"""
        return (1 - material.graphene_content/100) * 1e12 * material.degradation_rate
    
    def _generate_recommendations(self, material: CarbonMaterial) -> List[str]:
        """Generate recommendations for material improvement"""
        recommendations = []
        
        if material.graphene_content < 90:
            recommendations.append("Increase graphene content to improve conductivity")
        if material.surface_area < 1000:
            recommendations.append("Optimize synthesis to increase surface area")
        if material.degradation_rate > 0.2:
            recommendations.append("Investigate protective coatings to reduce degradation")
            
        return recommendations 