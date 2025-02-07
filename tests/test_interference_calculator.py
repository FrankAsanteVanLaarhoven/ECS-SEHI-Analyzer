import pytest
import numpy as np
from src.ecs_sehi_analyzer.core.hologram.interference_calculator import (
    InterferenceCalculator,
    InterferenceConfig,
    InterferenceType
)

@pytest.fixture
def calculator():
    return InterferenceCalculator()

@pytest.fixture
def sample_waves():
    size = 32  # Smaller size for testing
    wave1 = np.exp(1j * np.random.random((size, size)))
    wave2 = np.exp(1j * np.random.random((size, size)))
    return wave1, wave2

def test_interference_calculation(calculator, sample_waves):
    wave1, wave2 = sample_waves
    result = calculator.calculate_interference(wave1, wave2)
    
    assert result["success"]
    assert "intensity" in result
    assert "contrast" in result
    assert "interference_type" in result
    assert 0 <= result["contrast"] <= 1

def test_wave_propagation(calculator, sample_waves):
    wave1, _ = sample_waves
    result = calculator.propagate_wave(wave1)
    
    assert result["success"]
    assert "wave" in result
    assert result["wave"].shape == wave1.shape
    assert "distance" in result

def test_coherence_calculation(calculator, sample_waves):
    wave1, wave2 = sample_waves
    result = calculator.calculate_coherence(wave1, wave2)
    
    assert result["success"]
    assert "coherence" in result
    assert "phase" in result
    assert 0 <= result["coherence"] <= 1

def test_constructive_interference(calculator):
    size = 32
    # Create identical waves for constructive interference
    wave = np.exp(1j * np.random.random((size, size)))
    result = calculator.calculate_interference(wave, wave)
    
    assert result["success"]
    assert result["interference_type"] == InterferenceType.CONSTRUCTIVE.value
    assert result["contrast"] > 0.8

def test_destructive_interference(calculator):
    size = 32
    wave = np.exp(1j * np.random.random((size, size)))
    # Create opposite phase for destructive interference
    anti_wave = -wave
    result = calculator.calculate_interference(wave, anti_wave)
    
    assert result["success"]
    assert result["interference_type"] == InterferenceType.DESTRUCTIVE.value
    assert result["contrast"] < 0.2

def test_mismatched_wave_shapes(calculator, sample_waves):
    wave1, _ = sample_waves
    wave2 = np.exp(1j * np.random.random((16, 16)))  # Different size
    result = calculator.calculate_interference(wave1, wave2)
    
    assert not result["success"]
    assert "error" in result

def test_config_options():
    config = InterferenceConfig(
        wavelength=532e-9,  # Green laser
        resolution=(512, 512),
        propagation_distance=0.2
    )
    calculator = InterferenceCalculator(config)
    
    assert calculator.config.wavelength == 532e-9
    assert calculator.config.resolution == (512, 512)
    assert calculator.config.propagation_distance == 0.2

def test_phase_difference_effect(calculator, sample_waves):
    wave1, wave2 = sample_waves
    
    # Test different phase differences
    phase_diffs = [0, np.pi/2, np.pi]
    results = []
    
    for phase in phase_diffs:
        result = calculator.calculate_interference(wave1, wave2, phase)
        results.append(result)
        
    # Patterns should be different for different phases
    assert len(set(r["interference_type"] for r in results)) > 1