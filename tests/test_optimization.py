import pytest
import numpy as np
from src.ecs_sehi_analyzer.core.quantum.optimization import (
    QuantumOptimizer,
    OptimizationConfig,
    OptimizationMethod
)

@pytest.fixture
def optimizer():
    return QuantumOptimizer()

@pytest.fixture
def sample_objective():
    def objective(params):
        return np.sum(params**2)  # Simple quadratic function
    return objective

def test_gradient_descent(optimizer, sample_objective):
    initial_params = np.array([1.0, 1.0])
    
    result = optimizer.optimize(
        sample_objective,
        initial_params,
        OptimizationMethod.GRADIENT_DESCENT
    )
    
    assert result["success"]
    assert "optimal_params" in result
    assert "optimal_value" in result
    assert result["optimal_value"] < sample_objective(initial_params)

def test_optimization_convergence(optimizer, sample_objective):
    initial_params = np.array([0.1, 0.1])
    
    result = optimizer.optimize(
        sample_objective,
        initial_params,
        OptimizationMethod.GRADIENT_DESCENT
    )
    
    assert result["success"]
    assert result["convergence"]
    assert result["iterations"] < optimizer.config.max_iterations

def test_gradient_computation(optimizer, sample_objective):
    params = np.array([1.0, 1.0])
    
    gradient = optimizer._compute_gradient(sample_objective, params)
    
    assert isinstance(gradient, np.ndarray)
    assert gradient.shape == params.shape
    # For quadratic function, gradient should point towards origin
    assert np.all(gradient > 0)

def test_optimization_history(optimizer, sample_objective):
    initial_params = np.array([1.0, 1.0])
    
    optimizer.optimize(
        sample_objective,
        initial_params,
        OptimizationMethod.GRADIENT_DESCENT
    )
    
    assert len(optimizer.optimization_history) > 0
    for record in optimizer.optimization_history:
        assert "iteration" in record
        assert "value" in record
        assert "params" in record

def test_invalid_method(optimizer, sample_objective):
    initial_params = np.array([1.0, 1.0])
    
    with pytest.raises(ValueError):
        optimizer.optimize(
            sample_objective,
            initial_params,
            "invalid_method"
        )

def test_config_options():
    config = OptimizationConfig(
        max_iterations=500,
        convergence_threshold=1e-8,
        learning_rate=0.05
    )
    optimizer = QuantumOptimizer(config)
    
    assert optimizer.config.max_iterations == 500
    assert optimizer.config.convergence_threshold == 1e-8
    assert optimizer.config.learning_rate == 0.05 