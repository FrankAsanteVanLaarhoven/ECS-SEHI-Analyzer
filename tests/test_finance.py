import pytest
import numpy as np
from datetime import datetime
from src.ecs_sehi_analyzer.core.quantum.finance import (
    QuantumFinanceEngine,
    FinanceConfig,
    PortfolioStrategy,
    RiskModel
)

@pytest.fixture
def engine():
    return QuantumFinanceEngine()

@pytest.fixture
def sample_data():
    num_assets = 4
    time_horizon = 252
    
    # Generate sample returns and covariance
    returns = np.random.normal(0.001, 0.02, (num_assets, time_horizon))
    covariance = np.cov(returns)
    
    return returns, covariance

def test_portfolio_optimization(engine, sample_data):
    returns, covariance = sample_data
    
    for strategy in PortfolioStrategy:
        result = engine.optimize_portfolio(returns, covariance, strategy)
        
        assert result["success"]
        assert "weights" in result
        assert "expected_return" in result
        assert "risk" in result
        assert "metrics" in result
        
        weights = result["weights"]
        assert len(weights) == len(returns)
        assert np.abs(np.sum(weights) - 1.0) < 1e-6  # Weights sum to 1
        assert np.all(weights >= 0)  # Non-negative weights

def test_minimum_risk_strategy(engine, sample_data):
    returns, covariance = sample_data
    
    result = engine.optimize_portfolio(
        returns,
        covariance,
        PortfolioStrategy.MINIMUM_RISK
    )
    
    assert result["success"]
    assert "diversification" in result["metrics"]
    assert result["metrics"]["diversification"] > 0

def test_maximum_return_strategy(engine, sample_data):
    returns, covariance = sample_data
    
    result = engine.optimize_portfolio(
        returns,
        covariance,
        PortfolioStrategy.MAXIMUM_RETURN
    )
    
    assert result["success"]
    assert "concentration" in result["metrics"]
    assert 0 <= result["metrics"]["concentration"] <= 1

def test_balanced_strategy(engine, sample_data):
    returns, covariance = sample_data
    
    result = engine.optimize_portfolio(
        returns,
        covariance,
        PortfolioStrategy.BALANCED
    )
    
    assert result["success"]
    assert "balance_score" in result["metrics"]

def test_quantum_enhanced_strategy(engine, sample_data):
    returns, covariance = sample_data
    
    result = engine.optimize_portfolio(
        returns,
        covariance,
        PortfolioStrategy.QUANTUM_ENHANCED
    )
    
    assert result["success"]
    assert "quantum_advantage" in result["metrics"]
    assert 0 <= result["metrics"]["quantum_advantage"] <= 1

def test_risk_estimation(engine, sample_data):
    returns, _ = sample_data
    
    for model in RiskModel:
        result = engine.estimate_risk(returns, model)
        
        assert result["success"]
        assert "risk" in result
        assert "confidence" in result
        assert result["model"] == model.value
        assert 0 <= result["confidence"] <= 1

def test_classical_risk_estimation(engine, sample_data):
    returns, _ = sample_data
    
    result = engine.estimate_risk(returns, RiskModel.CLASSICAL)
    assert result["success"]
    assert result["confidence"] == 0.95  # Classical confidence level

def test_quantum_risk_estimation(engine, sample_data):
    returns, _ = sample_data
    
    result = engine.estimate_risk(returns, RiskModel.QUANTUM)
    assert result["success"]
    assert result["confidence"] > 0.9  # High confidence from quantum estimation

def test_hybrid_risk_estimation(engine, sample_data):
    returns, _ = sample_data
    
    result = engine.estimate_risk(returns, RiskModel.HYBRID)
    assert result["success"]
    
    # Hybrid should have confidence at least as good as classical
    assert result["confidence"] >= 0.95

def test_invalid_data(engine):
    # Test mismatched dimensions
    returns = np.random.normal(0, 1, (5, 100))
    covariance = np.random.normal(0, 1, (4, 4))
    
    result = engine.optimize_portfolio(
        returns,
        covariance,
        PortfolioStrategy.MINIMUM_RISK
    )
    
    assert not result["success"]
    assert "error" in result

def test_optimization_history(engine, sample_data):
    returns, covariance = sample_data
    
    # Perform multiple optimizations
    for strategy in PortfolioStrategy:
        engine.optimize_portfolio(returns, covariance, strategy)
    
    assert len(engine.optimization_history) == len(PortfolioStrategy)
    
    for record in engine.optimization_history:
        assert "timestamp" in record
        assert "strategy" in record
        assert "num_assets" in record
        assert "result" in record

def test_config_options():
    config = FinanceConfig(
        num_assets=20,
        time_horizon=126,
        risk_tolerance=0.3
    )
    engine = QuantumFinanceEngine(config)
    
    assert engine.config.num_assets == 20
    assert engine.config.time_horizon == 126
    assert engine.config.risk_tolerance == 0.3 