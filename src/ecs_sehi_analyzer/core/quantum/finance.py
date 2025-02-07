from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum
import streamlit as st
from datetime import datetime
from .circuit import QuantumCircuitEngine, GateType
from .optimization import QuantumOptimizer, OptimizationConfig

class PortfolioStrategy(Enum):
    MINIMUM_RISK = "minimum_risk"
    MAXIMUM_RETURN = "maximum_return"
    BALANCED = "balanced"
    QUANTUM_ENHANCED = "quantum_enhanced"

class RiskModel(Enum):
    CLASSICAL = "classical"
    QUANTUM = "quantum"
    HYBRID = "hybrid"

@dataclass
class FinanceConfig:
    """Quantum finance configuration"""
    num_assets: int = 10
    time_horizon: int = 252  # Trading days
    risk_tolerance: float = 0.5  # 0 to 1
    rebalance_frequency: int = 21  # Days
    quantum_shots: int = 1000
    metadata: Dict = field(default_factory=dict)

class QuantumFinanceEngine:
    def __init__(self, config: Optional[FinanceConfig] = None):
        self.config = config or FinanceConfig()
        self.circuit = QuantumCircuitEngine()
        self.optimizer = QuantumOptimizer()
        
        self.portfolios: Dict[str, Dict] = {}
        self.risk_models: Dict[str, Dict] = {}
        self.optimization_history: List[Dict] = []
        
    def optimize_portfolio(self,
                         returns: np.ndarray,
                         covariance: np.ndarray,
                         strategy: PortfolioStrategy,
                         constraints: Optional[Dict] = None) -> Dict:
        """Optimize portfolio using quantum algorithms"""
        try:
            if returns.shape[0] != covariance.shape[0]:
                raise ValueError("Returns and covariance dimensions mismatch")
                
            # Initialize quantum circuit for optimization
            self.circuit.initialize_circuit()
            
            # Setup optimization based on strategy
            if strategy == PortfolioStrategy.MINIMUM_RISK:
                result = self._minimize_risk(returns, covariance, constraints)
            elif strategy == PortfolioStrategy.MAXIMUM_RETURN:
                result = self._maximize_return(returns, covariance, constraints)
            elif strategy == PortfolioStrategy.BALANCED:
                result = self._balanced_optimization(returns, covariance, constraints)
            elif strategy == PortfolioStrategy.QUANTUM_ENHANCED:
                result = self._quantum_enhanced_optimization(returns, covariance, constraints)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
            # Record optimization
            optimization = {
                "timestamp": datetime.now(),
                "strategy": strategy.value,
                "num_assets": len(returns),
                "result": result
            }
            self.optimization_history.append(optimization)
            
            return {
                "success": True,
                "weights": result["weights"],
                "expected_return": result["expected_return"],
                "risk": result["risk"],
                "metrics": result["metrics"]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def estimate_risk(self,
                     returns: np.ndarray,
                     model: RiskModel = RiskModel.QUANTUM) -> Dict:
        """Estimate portfolio risk using quantum algorithms"""
        try:
            if model == RiskModel.CLASSICAL:
                risk = self._classical_risk_estimation(returns)
            elif model == RiskModel.QUANTUM:
                risk = self._quantum_risk_estimation(returns)
            elif model == RiskModel.HYBRID:
                risk = self._hybrid_risk_estimation(returns)
            else:
                raise ValueError(f"Unknown risk model: {model}")
                
            return {
                "success": True,
                "risk": risk["value"],
                "confidence": risk["confidence"],
                "model": model.value
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def render_finance_interface(self):
        """Render Streamlit finance interface"""
        st.markdown("### 💹 Quantum Finance")
        
        # Portfolio optimization
        st.markdown("#### Portfolio Optimization")
        
        # Input data
        st.markdown("##### Asset Returns")
        num_assets = st.number_input(
            "Number of Assets",
            min_value=2,
            max_value=50,
            value=self.config.num_assets
        )
        
        # Generate sample data if needed
        if st.button("Generate Sample Data"):
            returns = np.random.normal(0.001, 0.02, (num_assets, self.config.time_horizon))
            covariance = np.cov(returns)
            
            st.session_state["returns"] = returns
            st.session_state["covariance"] = covariance
            
        # Strategy selection
        strategy = st.selectbox(
            "Optimization Strategy",
            [s.value for s in PortfolioStrategy]
        )
        
        # Optimize button
        if st.button("Optimize Portfolio"):
            if "returns" not in st.session_state:
                st.error("Please generate sample data first")
            else:
                result = self.optimize_portfolio(
                    st.session_state["returns"],
                    st.session_state["covariance"],
                    PortfolioStrategy(strategy)
                )
                
                if result["success"]:
                    self._render_optimization_result(result)
                else:
                    st.error(f"Optimization failed: {result.get('error')}")
                    
    def _minimize_risk(self, returns: np.ndarray, covariance: np.ndarray, constraints: Dict) -> Dict:
        """Minimize portfolio risk"""
        # Setup quantum optimization problem
        def objective(weights):
            return np.dot(weights, np.dot(covariance, weights))
            
        result = self.optimizer.optimize(
            parameter_space={"weights": (0, 1)},
            fitness_function=lambda p: -objective(p["weights"])
        )
        
        weights = result.parameters["weights"]
        expected_return = np.dot(weights, np.mean(returns, axis=1))
        risk = np.sqrt(objective(weights))
        
        return {
            "weights": weights,
            "expected_return": expected_return,
            "risk": risk,
            "metrics": {
                "sharpe_ratio": expected_return / risk if risk > 0 else 0,
                "diversification": 1 - np.sum(weights**2)
            }
        }
        
    def _maximize_return(self, returns: np.ndarray, covariance: np.ndarray, constraints: Dict) -> Dict:
        """Maximize portfolio return"""
        # Setup quantum optimization problem
        def objective(weights):
            return np.dot(weights, np.mean(returns, axis=1))
            
        result = self.optimizer.optimize(
            parameter_space={"weights": (0, 1)},
            fitness_function=objective
        )
        
        weights = result.parameters["weights"]
        expected_return = objective(weights)
        risk = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
        
        return {
            "weights": weights,
            "expected_return": expected_return,
            "risk": risk,
            "metrics": {
                "sharpe_ratio": expected_return / risk if risk > 0 else 0,
                "concentration": np.max(weights)
            }
        }
        
    def _balanced_optimization(self, returns: np.ndarray, covariance: np.ndarray, constraints: Dict) -> Dict:
        """Balanced risk-return optimization"""
        # Combine risk and return objectives
        def objective(weights):
            ret = np.dot(weights, np.mean(returns, axis=1))
            risk = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
            return ret - self.config.risk_tolerance * risk
            
        result = self.optimizer.optimize(
            parameter_space={"weights": (0, 1)},
            fitness_function=objective
        )
        
        weights = result.parameters["weights"]
        expected_return = np.dot(weights, np.mean(returns, axis=1))
        risk = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
        
        return {
            "weights": weights,
            "expected_return": expected_return,
            "risk": risk,
            "metrics": {
                "sharpe_ratio": expected_return / risk if risk > 0 else 0,
                "balance_score": objective(weights)
            }
        }
        
    def _quantum_enhanced_optimization(self, returns: np.ndarray, covariance: np.ndarray, constraints: Dict) -> Dict:
        """Quantum-enhanced portfolio optimization"""
        # Create quantum circuit for optimization
        num_qubits = int(np.ceil(np.log2(len(returns))))
        
        # Initialize quantum state
        for i in range(num_qubits):
            self.circuit.add_gate(GateType.HADAMARD, [i])
            
        # Add quantum operations for optimization
        for i in range(num_qubits):
            phase = np.arcsin(np.sqrt(returns[i]))
            self.circuit.add_gate(GateType.PHASE, [i], [phase])
            
        # Measure
        result = self.circuit.execute_circuit()
        
        # Process quantum measurements
        counts = result["counts"]
        total_shots = sum(counts.values())
        
        # Convert measurements to weights
        weights = np.zeros(len(returns))
        for state, count in counts.items():
            idx = int(state, 2)
            if idx < len(returns):
                weights[idx] = count / total_shots
                
        # Normalize weights
        weights = weights / np.sum(weights)
        
        expected_return = np.dot(weights, np.mean(returns, axis=1))
        risk = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
        
        return {
            "weights": weights,
            "expected_return": expected_return,
            "risk": risk,
            "metrics": {
                "sharpe_ratio": expected_return / risk if risk > 0 else 0,
                "quantum_advantage": len(counts) / len(returns)
            }
        }
        
    def _classical_risk_estimation(self, returns: np.ndarray) -> Dict:
        """Classical risk estimation"""
        volatility = np.std(returns)
        return {
            "value": volatility,
            "confidence": 0.95
        }
        
    def _quantum_risk_estimation(self, returns: np.ndarray) -> Dict:
        """Quantum risk estimation"""
        # Create quantum circuit for risk estimation
        self.circuit.initialize_circuit()
        
        # Encode returns into quantum state
        normalized_returns = returns / np.linalg.norm(returns)
        self.circuit.circuit.initialize(normalized_returns, self.circuit.circuit.qubits)
        
        # Add quantum operations
        self.circuit.add_gate(GateType.HADAMARD, [0])
        self.circuit.add_gate(GateType.PHASE, [0], [np.pi/4])
        
        # Measure
        result = self.circuit.execute_circuit()
        
        # Process results
        counts = result["counts"]
        total_shots = sum(counts.values())
        p0 = counts.get("0", 0) / total_shots
        
        risk = np.sqrt(-2 * np.log(p0)) * np.std(returns)
        confidence = 1 - (1 / total_shots)
        
        return {
            "value": risk,
            "confidence": confidence
        }
        
    def _hybrid_risk_estimation(self, returns: np.ndarray) -> Dict:
        """Hybrid classical-quantum risk estimation"""
        # Combine classical and quantum estimates
        classical = self._classical_risk_estimation(returns)
        quantum = self._quantum_risk_estimation(returns)
        
        # Weight estimates based on confidence
        total_confidence = classical["confidence"] + quantum["confidence"]
        weighted_risk = (
            classical["value"] * classical["confidence"] +
            quantum["value"] * quantum["confidence"]
        ) / total_confidence
        
        return {
            "value": weighted_risk,
            "confidence": np.maximum(classical["confidence"], quantum["confidence"])
        }
        
    def _render_optimization_result(self, result: Dict):
        """Render optimization result visualization"""
        import plotly.graph_objects as go
        
        # Portfolio weights
        fig = go.Figure(data=[
            go.Bar(
                x=[f"Asset {i+1}" for i in range(len(result["weights"]))],
                y=result["weights"]
            )
        ])
        
        fig.update_layout(
            title="Portfolio Weights",
            xaxis_title="Asset",
            yaxis_title="Weight",
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Expected Return",
                f"{result['expected_return']:.2%}"
            )
        with col2:
            st.metric(
                "Risk",
                f"{result['risk']:.2%}"
            )
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{result['metrics']['sharpe_ratio']:.2f}"
            ) 