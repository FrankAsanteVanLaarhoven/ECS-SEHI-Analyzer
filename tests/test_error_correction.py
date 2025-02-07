import pytest
import numpy as np
from src.ecs_sehi_analyzer.core.quantum.error_correction import (
    QuantumErrorCorrector,
    CorrectionConfig,
    CorrectionCode,
    ErrorType
)

@pytest.fixture
def corrector():
    return QuantumErrorCorrector()

@pytest.fixture
def sample_state():
    # Create normalized single-qubit state
    return np.array([1/np.sqrt(2), 1/np.sqrt(2)])

def test_bit_flip_encoding(corrector, sample_state):
    result = corrector.encode_state(
        sample_state,
        CorrectionCode.BIT_FLIP
    )
    
    assert result["success"]
    assert "encoded_state" in result
    assert "ancilla_qubits" in result
    assert "logical_mapping" in result
    assert len(result["ancilla_qubits"]) == 2  # Two ancilla qubits for bit-flip code

def test_phase_flip_encoding(corrector, sample_state):
    result = corrector.encode_state(
        sample_state,
        CorrectionCode.PHASE_FLIP
    )
    
    assert result["success"]
    assert "encoded_state" in result
    assert len(result["encoded_state"]) == 8  # 3-qubit encoding

def test_error_detection(corrector, sample_state):
    # First encode the state
    encoded = corrector.encode_state(
        sample_state,
        CorrectionCode.BIT_FLIP
    )
    
    # Then detect errors
    result = corrector.detect_errors(
        encoded["encoded_state"],
        ErrorType.BIT
    )
    
    assert result["success"]
    assert "error_detected" in result
    assert "error_locations" in result
    assert "syndrome_data" in result

def test_error_correction(corrector, sample_state):
    # Encode state
    encoded = corrector.encode_state(
        sample_state,
        CorrectionCode.BIT_FLIP
    )
    
    # Detect errors
    error_data = corrector.detect_errors(
        encoded["encoded_state"],
        ErrorType.BIT
    )
    
    # Correct errors
    result = corrector.correct_errors(
        encoded["encoded_state"],
        error_data,
        CorrectionCode.BIT_FLIP
    )
    
    assert result["success"]
    assert "corrected_state" in result
    assert "fidelity" in result
    assert 0 <= result["fidelity"] <= 1

def test_syndrome_measurement(corrector, sample_state):
    # Encode state
    encoded = corrector.encode_state(
        sample_state,
        CorrectionCode.BIT_FLIP
    )
    
    # Measure syndrome
    syndrome = corrector._measure_bit_syndrome(encoded["encoded_state"])
    
    assert isinstance(syndrome, np.ndarray)
    assert len(syndrome) == 2  # Two syndrome bits for bit-flip code

def test_syndrome_analysis(corrector):
    # Create sample syndrome data
    syndromes = np.array([
        [1, 0],
        [1, 0],
        [1, 0]
    ])
    
    error_locations = corrector._analyze_syndromes(syndromes)
    assert isinstance(error_locations, list)
    assert all(isinstance(x, int) for x in error_locations)

def test_correction_history(corrector, sample_state):
    # Perform multiple encodings
    for code in [CorrectionCode.BIT_FLIP, CorrectionCode.PHASE_FLIP]:
        corrector.encode_state(sample_state, code)
    
    assert len(corrector.correction_history) == 2
    
    for record in corrector.correction_history:
        assert "timestamp" in record
        assert "code" in record
        assert "state_size" in record
        assert "result" in record

def test_invalid_code(corrector, sample_state):
    with pytest.raises(ValueError):
        corrector.encode_state(sample_state, "invalid_code")

def test_invalid_error_type(corrector, sample_state):
    encoded = corrector.encode_state(
        sample_state,
        CorrectionCode.BIT_FLIP
    )
    
    with pytest.raises(ValueError):
        corrector.detect_errors(
            encoded["encoded_state"],
            "invalid_error_type"
        )

def test_config_options():
    config = CorrectionConfig(
        code_distance=5,
        measurement_rounds=10,
        error_threshold=0.005,
        syndrome_method="fault_tolerant"
    )
    corrector = QuantumErrorCorrector(config)
    
    assert corrector.config.code_distance == 5
    assert corrector.config.measurement_rounds == 10
    assert corrector.config.error_threshold == 0.005
    assert corrector.config.syndrome_method == "fault_tolerant"

def test_logical_state_preservation(corrector, sample_state):
    # Encode state
    encoded = corrector.encode_state(
        sample_state,
        CorrectionCode.BIT_FLIP
    )
    
    # Verify logical state is preserved
    logical_state = encoded["encoded_state"][::3]  # Extract logical state
    fidelity = np.abs(np.vdot(sample_state, logical_state))**2
    assert fidelity > 0.99  # High fidelity with original state 