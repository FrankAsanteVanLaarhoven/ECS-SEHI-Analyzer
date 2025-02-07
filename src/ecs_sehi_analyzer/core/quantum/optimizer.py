from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from enum import Enum
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
from ..neural.material_network import NetworkConfig, MaterialAnalysisNetwork

class OptimizationMetric(Enum):
    LOSS = "loss"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    CONVERGENCE = "convergence"
    RESOURCE_USAGE = "resource_usage"

@dataclass
class OptimizationConfig:
    """Quantum optimization configuration"""
    population_size: int = 50
    generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    elite_size: int = 5
    target_metric: OptimizationMetric = OptimizationMetric.ACCURACY
    timeout: float = 3600  # seconds
    metadata: Dict = field(default_factory=dict)

@dataclass
class OptimizationResult:
    """Optimization result"""
    parameters: Dict
    fitness: float
    generation: int
    timestamp: datetime
    metrics: Dict[str, float]
    convergence_history: List[float]

class QuantumOptimizer:
    def __init__(self, config: Optional[OptimizationConfig] = None):
        self.config = config or OptimizationConfig()
        self.optimization_history: List[OptimizationResult] = []
        self.best_result: Optional[OptimizationResult] = None
        self.fitness_function: Optional[Callable] = None
        
    def optimize(self, 
                parameter_space: Dict[str, Tuple[float, float]],
                fitness_function: Callable) -> OptimizationResult:
        """Run quantum optimization"""
        self.fitness_function = fitness_function
        
        # Initialize population
        population = self._initialize_population(parameter_space)
        convergence_history = []
        
        for generation in range(self.config.generations):
            # Evaluate fitness
            fitness_scores = [
                self._evaluate_individual(individual)
                for individual in population
            ]
            
            # Track best result
            best_idx = np.argmax(fitness_scores)
            best_fitness = fitness_scores[best_idx]
            convergence_history.append(best_fitness)
            
            # Create new result
            result = OptimizationResult(
                parameters=population[best_idx],
                fitness=best_fitness,
                generation=generation,
                timestamp=datetime.now(),
                metrics=self._calculate_metrics(population, fitness_scores),
                convergence_history=convergence_history
            )
            
            # Update history
            self.optimization_history.append(result)
            if not self.best_result or result.fitness > self.best_result.fitness:
                self.best_result = result
                
            # Check convergence
            if self._check_convergence(convergence_history):
                break
                
            # Create next generation
            population = self._create_next_generation(
                population,
                fitness_scores
            )
            
        return self.best_result
        
    def render_optimization_dashboard(self):
        """Render Streamlit optimization dashboard"""
        st.markdown("### 🎯 Quantum Optimization")
        
        # Current status
        if self.best_result:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Best Fitness",
                    f"{self.best_result.fitness:.4f}"
                )
                
            with col2:
                st.metric(
                    "Generation",
                    self.best_result.generation
                )
                
            # Convergence plot
            st.markdown("#### Convergence History")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                y=self.best_result.convergence_history,
                mode="lines",
                name="Fitness"
            ))
            
            fig.update_layout(
                title="Optimization Convergence",
                xaxis_title="Generation",
                yaxis_title="Fitness"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Best parameters
            st.markdown("#### Best Parameters")
            st.json(self.best_result.parameters)
            
    def _initialize_population(self, 
                             parameter_space: Dict[str, Tuple[float, float]]) -> List[Dict]:
        """Initialize random population"""
        population = []
        
        for _ in range(self.config.population_size):
            individual = {}
            for param, (min_val, max_val) in parameter_space.items():
                individual[param] = np.random.uniform(min_val, max_val)
            population.append(individual)
            
        return population
        
    def _evaluate_individual(self, individual: Dict) -> float:
        """Evaluate individual's fitness"""
        if not self.fitness_function:
            raise ValueError("Fitness function not set")
            
        try:
            return self.fitness_function(individual)
        except Exception as e:
            st.error(f"Fitness evaluation failed: {str(e)}")
            return float("-inf")
            
    def _create_next_generation(self, 
                              population: List[Dict],
                              fitness_scores: List[float]) -> List[Dict]:
        """Create next generation using quantum-inspired evolution"""
        next_generation = []
        
        # Elitism
        elite_indices = np.argsort(fitness_scores)[-self.config.elite_size:]
        next_generation.extend([population[i] for i in elite_indices])
        
        # Quantum crossover and mutation
        while len(next_generation) < self.config.population_size:
            if np.random.random() < self.config.crossover_rate:
                # Select parents
                parent1, parent2 = self._select_parents(population, fitness_scores)
                child = self._quantum_crossover(parent1, parent2)
            else:
                # Select individual for mutation
                individual = self._select_parents(population, fitness_scores)[0]
                child = self._quantum_mutation(individual)
                
            next_generation.append(child)
            
        return next_generation[:self.config.population_size]
        
    def _select_parents(self, 
                       population: List[Dict],
                       fitness_scores: List[float]) -> Tuple[Dict, Dict]:
        """Select parents using tournament selection"""
        tournament_size = 3
        selected = []
        
        for _ in range(2):
            indices = np.random.choice(
                len(population),
                tournament_size,
                replace=False
            )
            tournament_fitness = [fitness_scores[i] for i in indices]
            winner_idx = indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
            
        return tuple(selected)
        
    def _quantum_crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Perform quantum-inspired crossover"""
        child = {}
        
        for param in parent1.keys():
            if np.random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
                
        return child
        
    def _quantum_mutation(self, individual: Dict) -> Dict:
        """Perform quantum-inspired mutation"""
        mutated = individual.copy()
        
        for param in mutated:
            if np.random.random() < self.config.mutation_rate:
                # Apply quantum noise
                noise = np.random.normal(0, 0.1)
                mutated[param] += noise
                
        return mutated
        
    def _check_convergence(self, history: List[float]) -> bool:
        """Check if optimization has converged"""
        if len(history) < 10:
            return False
            
        recent_improvement = history[-1] - history[-10]
        return abs(recent_improvement) < 1e-6
        
    def _calculate_metrics(self, 
                         population: List[Dict],
                         fitness_scores: List[float]) -> Dict[str, float]:
        """Calculate optimization metrics"""
        return {
            "mean_fitness": np.mean(fitness_scores),
            "std_fitness": np.std(fitness_scores),
            "population_diversity": self._calculate_diversity(population),
            "improvement_rate": self._calculate_improvement_rate()
        }
        
    def _calculate_diversity(self, population: List[Dict]) -> float:
        """Calculate population diversity"""
        param_values = []
        for param in population[0].keys():
            values = [ind[param] for ind in population]
            param_values.extend(values)
            
        return np.std(param_values)
        
    def _calculate_improvement_rate(self) -> float:
        """Calculate improvement rate"""
        if len(self.optimization_history) < 2:
            return 0.0
            
        recent = self.optimization_history[-1].fitness
        previous = self.optimization_history[-2].fitness
        
        return (recent - previous) / previous if previous != 0 else 0.0 