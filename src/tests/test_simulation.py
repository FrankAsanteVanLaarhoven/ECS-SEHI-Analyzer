import pytest
import numpy as np
from app.utils.simulation import SEHISimulator, SimulationParameters

class TestSEHISimulator:
    def test_simulation_parameters(self):
        """Test simulation parameters validation."""
        params = SimulationParameters(
            temperature=25.0,
            humidity=50.0,
            stress_level=0.5,
            time_steps=100
        )
        assert params.temperature == 25.0
        assert params.humidity == 50.0
        
    def test_degradation_simulation(self):
        """Test degradation simulation."""
        simulator = SEHISimulator()
        initial_state = np.ones((10, 10))
        
        results = simulator.simulate_degradation(initial_state)
        
        assert results is not None
        assert 'final_state' in results
        assert 'history' in results
        assert results['history'].shape[0] == 100  # Default time steps
        
    def test_visualization(self):
        """Test simulation visualization."""
        simulator = SEHISimulator()
        initial_state = np.ones((10, 10))
        results = simulator.simulate_degradation(initial_state)
        
        fig = simulator.visualize_simulation(results)
        assert fig is not None 