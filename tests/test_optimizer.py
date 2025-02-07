import pytest
import numpy as np
from datetime import datetime
from src.ecs_sehi_analyzer.core.quantum.optimizer import (
    QuantumOptimizer,
    OptimizationConfig,
    OptimizationMetric,
    OptimizationResult
)

@pytest.fixture
def optimizer():
    return QuantumOptimizer()

@pytest.fixture
def parameter_space():
    return {
        "learning_rate": (0.0001, 0.1),
        "batch_size": (16, 128),
        "hidden_size": (64, 512)
    }

def test_initialization(optimizer):
    assert isinstance(optimizer.config, OptimizationConfig)
    assert len(optimizer.optimization_history) == 0
    assert optimizer.best_result is None

def test_population_initialization(optimizer, parameter_space):
    population = optimizer._initialize_population(parameter_space)
    
    assert len(population) == optimizer.config.population_size
    for individual in population:
        assert set(individual.keys()) == set(parameter_space.keys())
        for param, (min_val, max_val) in parameter_space.items():
            assert min_val <= individual[param] <= max_val

def test_optimization(optimizer, parameter_space):
    # Simple quadratic fitness function
    def fitness_function(params):
        return -sum((x - 0.5)**2 for x in params.values())
    
    result = optimizer.optimize(parameter_space, fitness_function)
    
    assert isinstance(result, OptimizationResult)
    assert result.fitness == optimizer.best_result.fitness
    assert len(result.convergence_history) > 0

def test_quantum_operations(optimizer):
    parent1 = {"x": 0.1, "y": 0.2}
    parent2 = {"x": 0.3, "y": 0.4}
    
    # Test crossover
    child = optimizer._quantum_crossover(parent1, parent2)
    assert set(child.keys()) == set(parent1.keys())
    for param in child:
        assert child[param] in [parent1[param], parent2[param]]
    
    # Test mutation
    mutated = optimizer._quantum_mutation(parent1)
    assert set(mutated.keys()) == set(parent1.keys())
    assert any(mutated[k] != parent1[k] for k in parent1)

def test_convergence_check(optimizer):
    # Non-converged case
    history = [i * 0.1 for i in range(5)]
    assert not optimizer._check_convergence(history)
    
    # Converged case
    history = [1.0] * 10
    assert optimizer._check_convergence(history)

def test_metrics_calculation(optimizer):
    population = [
        {"x": 0.1, "y": 0.2},
        {"x": 0.3, "y": 0.4}
    ]
    fitness_scores = [0.5, 0.7]
    
    metrics = optimizer._calculate_metrics(population, fitness_scores)
    assert "mean_fitness" in metrics
    assert "std_fitness" in metrics
    assert "population_diversity" in metrics
    assert "improvement_rate" in metrics 