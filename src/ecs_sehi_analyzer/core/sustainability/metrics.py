from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from enum import Enum

class ImpactCategory(Enum):
    ENERGY = "energy"
    WATER = "water"
    WASTE = "waste"
    EMISSIONS = "emissions"
    MATERIALS = "materials"

@dataclass
class SustainabilityMetric:
    category: ImpactCategory
    value: float
    unit: str
    timestamp: datetime
    source: str
    confidence: float = 1.0
    metadata: Dict = field(default_factory=dict)

class SustainabilityEngine:
    def __init__(self):
        self.metrics: Dict[str, List[SustainabilityMetric]] = {}
        self.baselines: Dict[str, float] = {}
        self.targets: Dict[str, float] = {}
        self.reduction_strategies: Dict[str, List[str]] = {}
        
    def record_metric(self, project_id: str, metric: SustainabilityMetric):
        """Record sustainability metric"""
        if project_id not in self.metrics:
            self.metrics[project_id] = []
            
        self.metrics[project_id].append(metric)
        self._update_baselines(project_id)
        
    def set_target(self, category: ImpactCategory, target: float):
        """Set reduction target for impact category"""
        self.targets[category.value] = target
        
    def analyze_impact(self, project_id: str) -> Dict:
        """Analyze environmental impact"""
        if project_id not in self.metrics:
            return {"error": "No metrics found for project"}
            
        project_metrics = self.metrics[project_id]
        
        analysis = {
            "total_impact": self._calculate_total_impact(project_metrics),
            "trends": self._analyze_trends(project_metrics),
            "reduction_potential": self._calculate_reduction_potential(project_metrics),
            "recommendations": self._generate_recommendations(project_metrics)
        }
        
        return analysis
        
    def render_sustainability_dashboard(self):
        """Render Streamlit sustainability dashboard"""
        st.markdown("### 🌱 Sustainability Dashboard")
        
        # Overview metrics
        self._render_overview_metrics()
        
        # Impact trends
        st.markdown("#### Impact Trends")
        fig = self._create_impact_chart()
        st.plotly_chart(fig, use_container_width=True)
        
        # Project comparison
        if len(self.metrics) > 1:
            st.markdown("#### Project Comparison")
            fig = self._create_comparison_chart()
            st.plotly_chart(fig, use_container_width=True)
            
        # Reduction strategies
        st.markdown("#### Reduction Strategies")
        self._render_reduction_strategies()
        
    def _calculate_total_impact(self, metrics: List[SustainabilityMetric]) -> Dict:
        """Calculate total environmental impact"""
        totals = {}
        for metric in metrics:
            if metric.category.value not in totals:
                totals[metric.category.value] = 0
            totals[metric.category.value] += metric.value
            
        return totals
        
    def _analyze_trends(self, metrics: List[SustainabilityMetric]) -> Dict:
        """Analyze impact trends"""
        trends = {}
        for category in ImpactCategory:
            cat_metrics = [m for m in metrics if m.category == category]
            if cat_metrics:
                values = [m.value for m in cat_metrics]
                timestamps = [m.timestamp.timestamp() for m in cat_metrics]
                if len(values) > 1:
                    trend = np.polyfit(timestamps, values, 1)[0]
                    trends[category.value] = {
                        "slope": trend,
                        "direction": "increasing" if trend > 0 else "decreasing"
                    }
                    
        return trends
        
    def _calculate_reduction_potential(self, metrics: List[SustainabilityMetric]) -> Dict:
        """Calculate potential impact reductions"""
        potential = {}
        for category in ImpactCategory:
            cat_metrics = [m for m in metrics if m.category == category]
            if cat_metrics:
                current = sum(m.value for m in cat_metrics)
                target = self.targets.get(category.value)
                if target:
                    potential[category.value] = {
                        "current": current,
                        "target": target,
                        "reduction_needed": max(0, current - target)
                    }
                    
        return potential
        
    def _generate_recommendations(self, metrics: List[SustainabilityMetric]) -> List[str]:
        """Generate sustainability recommendations"""
        recommendations = []
        trends = self._analyze_trends(metrics)
        potential = self._calculate_reduction_potential(metrics)
        
        for category in ImpactCategory:
            if category.value in trends:
                trend = trends[category.value]
                if trend["direction"] == "increasing":
                    recommendations.append(
                        f"Implement reduction strategies for {category.value} - "
                        f"impact is increasing"
                    )
                    
            if category.value in potential:
                reduction = potential[category.value]["reduction_needed"]
                if reduction > 0:
                    recommendations.append(
                        f"Need to reduce {category.value} impact by "
                        f"{reduction:.1f} units to meet target"
                    )
                    
        return recommendations
        
    def _update_baselines(self, project_id: str):
        """Update impact baselines"""
        if project_id in self.metrics:
            for category in ImpactCategory:
                cat_metrics = [m for m in self.metrics[project_id] 
                             if m.category == category]
                if cat_metrics:
                    self.baselines[f"{project_id}_{category.value}"] = \
                        sum(m.value for m in cat_metrics[:3]) / 3
                        
    def _render_overview_metrics(self):
        """Render overview metrics"""
        total_metrics = sum(len(metrics) for metrics in self.metrics.values())
        total_projects = len(self.metrics)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Metrics", total_metrics)
        with col2:
            st.metric("Active Projects", total_projects)
        with col3:
            reduction = self._calculate_total_reduction()
            st.metric(
                "Impact Reduction",
                f"{reduction:.1%}",
                delta=f"{reduction*100:.1f}% vs baseline"
            )
            
    def _calculate_total_reduction(self) -> float:
        """Calculate total impact reduction"""
        if not self.baselines:
            return 0.0
            
        current_totals = {}
        for metrics in self.metrics.values():
            for metric in metrics[-10:]:  # Last 10 measurements
                if metric.category.value not in current_totals:
                    current_totals[metric.category.value] = []
                current_totals[metric.category.value].append(metric.value)
                
        reductions = []
        for category, values in current_totals.items():
            if values:
                current = np.mean(values)
                baseline = next(
                    (v for k, v in self.baselines.items() if category in k),
                    current
                )
                if baseline > 0:
                    reductions.append(1 - (current / baseline))
                    
        return np.mean(reductions) if reductions else 0.0
        
    def _create_impact_chart(self) -> go.Figure:
        """Create impact trend chart"""
        fig = go.Figure()
        
        for category in ImpactCategory:
            for project_id, metrics in self.metrics.items():
                cat_metrics = [m for m in metrics if m.category == category]
                if cat_metrics:
                    fig.add_trace(go.Scatter(
                        x=[m.timestamp for m in cat_metrics],
                        y=[m.value for m in cat_metrics],
                        name=f"{project_id} - {category.value}",
                        mode="lines+markers"
                    ))
                    
        fig.update_layout(
            title="Impact Trends Over Time",
            xaxis_title="Time",
            yaxis_title="Impact Value",
            showlegend=True
        )
        
        return fig
        
    def _create_comparison_chart(self) -> go.Figure:
        """Create project comparison chart"""
        fig = go.Figure()
        
        for project_id, metrics in self.metrics.items():
            total_impact = sum(m.value for m in metrics)
            fig.add_trace(go.Bar(
                name=project_id,
                x=[project_id],
                y=[total_impact],
                text=[f"{total_impact:.1f}"],
                textposition="auto"
            ))
            
        fig.update_layout(
            title="Total Impact by Project",
            xaxis_title="Project",
            yaxis_title="Total Impact",
            showlegend=False
        )
        
        return fig
        
    def _render_reduction_strategies(self):
        """Render reduction strategies interface"""
        for category in ImpactCategory:
            with st.expander(f"{category.value.title()} Reduction Strategies"):
                strategies = self.reduction_strategies.get(category.value, [])
                
                # Add new strategy
                new_strategy = st.text_input(
                    "New Strategy",
                    key=f"strategy_{category.value}"
                )
                if st.button("Add", key=f"add_{category.value}"):
                    if new_strategy:
                        if category.value not in self.reduction_strategies:
                            self.reduction_strategies[category.value] = []
                        self.reduction_strategies[category.value].append(new_strategy)
                        
                # Display existing strategies
                for strategy in strategies:
                    st.markdown(f"- {strategy}") 