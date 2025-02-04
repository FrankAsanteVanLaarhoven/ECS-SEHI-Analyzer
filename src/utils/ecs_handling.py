# utils/ecs_handling.py

class ECSMaterial:
    """
    Class to handle ECS material properties and degradation patterns.
    """
    def __init__(self, material_id, surface_chemistry, mechanical_stress_response, environmental_degradation):
        self.material_id = material_id
        self.surface_chemistry = surface_chemistry
        self.mechanical_stress_response = mechanical_stress_response
        self.environmental_degradation = environmental_degradation

    def get_degradation_score(self, heat, humidity, stress):
        """
        Calculate degradation score based on environmental conditions.
        
        Args:
            heat (float): Heat level.
            humidity (float): Humidity level.
            stress (float): Mechanical stress level.
        
        Returns:
            float: Degradation score.
        """
        degradation_score = (heat * 0.5) + (humidity * 0.3) + (stress * 0.2)
        return degradation_score

    def to_dict(self):
        """
        Convert ECS material properties to a dictionary.
        
        Returns:
            dict: Dictionary of ECS material properties.
        """
        return {
            "material_id": self.material_id,
            "surface_chemistry": self.surface_chemistry,
            "mechanical_stress_response": self.mechanical_stress_response,
            "environmental_degradation": self.environmental_degradation
        }