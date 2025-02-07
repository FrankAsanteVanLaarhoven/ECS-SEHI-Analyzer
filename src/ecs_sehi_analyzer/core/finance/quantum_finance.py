from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from enum import Enum
import streamlit as st
import plotly.graph_objects as go
from ..quantum.encryption import QuantumEncryptionEngine

class ResourceType(Enum):
    COMPUTE = "compute"
    MATERIALS = "materials"
    EQUIPMENT = "equipment"
    PERSONNEL = "personnel"
    SOFTWARE = "software"

@dataclass
class BudgetAllocation:
    resource_type: ResourceType
    amount: float
    start_date: datetime
    end_date: datetime
    project_id: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class ResourceUsage:
    resource_type: ResourceType
    amount_used: float
    timestamp: datetime
    user_id: str
    project_id: str
    efficiency_score: float = 1.0

class QuantumFinanceEngine:
    def __init__(self):
        self.encryption = QuantumEncryptionEngine()
        self.budgets: Dict[str, List[BudgetAllocation]] = {}
        self.usage_history: List[ResourceUsage] = []
        self.efficiency_metrics: Dict[str, float] = {}
        
    def allocate_budget(self, 
                       project_id: str,
                       allocations: List[BudgetAllocation]) -> Dict[str, float]:
        """Allocate budget using quantum optimization"""
        # Encrypt sensitive financial data
        encrypted_data = self.encryption.encrypt_data(
            str(allocations).encode(),
            f"budget_{project_id}"
        )
        
        # Store allocations
        self.budgets[project_id] = allocations
        
        # Calculate total per resource type
        totals = {}
        for alloc in allocations:
            if alloc.resource_type.value not in totals:
                totals[alloc.resource_type.value] = 0
            totals[alloc.resource_type.value] += alloc.amount
            
        return totals
        
    def record_usage(self, usage: ResourceUsage):
        """Record resource usage with efficiency tracking"""
        self.usage_history.append(usage)
        
        # Update efficiency metrics
        project_usages = [u for u in self.usage_history 
                         if u.project_id == usage.project_id]
        
        if project_usages:
            efficiency = np.mean([u.efficiency_score for u in project_usages])
            self.efficiency_metrics[usage.project_id] = efficiency
            
    def analyze_efficiency(self, project_id: str) -> Dict:
        """Analyze resource usage efficiency"""
        project_usages = [u for u in self.usage_history 
                         if u.project_id == project_id]
        
        if not project_usages:
            return {"error": "No usage data found"}
            
        analysis = {
            "efficiency_score": self.efficiency_metrics.get(project_id, 0),
            "resource_distribution": self._calculate_distribution(project_usages),
            "usage_trends": self._analyze_trends(project_usages),
            "recommendations": self._generate_recommendations(project_usages)
        }
        
        return analysis
    
    def render_finance_dashboard(self):
        """Render Streamlit finance dashboard"""
        st.markdown("### 💰 Quantum Finance Dashboard")
        
        # Overview metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_budget = sum(
                sum(a.amount for a in allocs)
                for allocs in self.budgets.values()
            )
            st.metric("Total Budget", f"${total_budget:,.2f}")
            
        with col2:
            total_used = sum(u.amount_used for u in self.usage_history)
            st.metric("Total Used", f"${total_used:,.2f}")
            
        with col3:
            avg_efficiency = np.mean(list(self.efficiency_metrics.values())) \
                if self.efficiency_metrics else 0
            st.metric("Avg Efficiency", f"{avg_efficiency:.1%}")
            
        # Resource allocation chart
        st.markdown("#### Resource Allocation")
        fig = self._create_allocation_chart()
        st.plotly_chart(fig, use_container_width=True)
        
        # Project efficiency comparison
        st.markdown("#### Project Efficiency Comparison")
        if self.efficiency_metrics:
            fig = self._create_efficiency_chart()
            st.plotly_chart(fig, use_container_width=True)
            
        # Resource management
        st.markdown("#### Resource Management")
        col1, col2 = st.columns(2)
        
        with col1:
            project_id = st.selectbox(
                "Select Project",
                list(self.budgets.keys()) if self.budgets else ["No projects"]
            )
            
        with col2:
            resource_type = st.selectbox(
                "Resource Type",
                [rt.value for rt in ResourceType]
            )
            
        if project_id in self.budgets:
            self._render_resource_management(project_id, ResourceType(resource_type))
            
    def _calculate_distribution(self, usages: List[ResourceUsage]) -> Dict:
        """Calculate resource distribution"""
        distribution = {}
        for usage in usages:
            if usage.resource_type.value not in distribution:
                distribution[usage.resource_type.value] = 0
            distribution[usage.resource_type.value] += usage.amount_used
        return distribution
        
    def _analyze_trends(self, usages: List[ResourceUsage]) -> Dict:
        """Analyze usage trends"""
        sorted_usages = sorted(usages, key=lambda u: u.timestamp)
        trends = {}
        
        for rt in ResourceType:
            rt_usages = [u.amount_used for u in sorted_usages 
                        if u.resource_type == rt]
            if rt_usages:
                trend = np.polyfit(range(len(rt_usages)), rt_usages, 1)[0]
                trends[rt.value] = trend
                
        return trends
        
    def _generate_recommendations(self, usages: List[ResourceUsage]) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        distribution = self._calculate_distribution(usages)
        trends = self._analyze_trends(usages)
        
        for rt, amount in distribution.items():
            trend = trends.get(rt, 0)
            if trend > 0:
                recommendations.append(
                    f"Consider optimizing {rt} usage - increasing trend detected"
                )
            elif amount > sum(distribution.values()) * 0.4:
                recommendations.append(
                    f"High {rt} usage detected - review allocation"
                )
                
        return recommendations
        
    def _create_allocation_chart(self) -> go.Figure:
        """Create resource allocation chart"""
        fig = go.Figure()
        
        for project_id, allocations in self.budgets.items():
            for rt in ResourceType:
                rt_allocs = [a.amount for a in allocations 
                            if a.resource_type == rt]
                if rt_allocs:
                    fig.add_trace(go.Bar(
                        name=rt.value,
                        x=[project_id],
                        y=[sum(rt_allocs)],
                        text=[f"${sum(rt_allocs):,.2f}"],
                        textposition="auto"
                    ))
                    
        fig.update_layout(
            barmode="stack",
            title="Resource Allocation by Project",
            xaxis_title="Project",
            yaxis_title="Amount ($)",
            showlegend=True
        )
        
        return fig
        
    def _create_efficiency_chart(self) -> go.Figure:
        """Create efficiency comparison chart"""
        fig = go.Figure()
        
        projects = list(self.efficiency_metrics.keys())
        efficiencies = [self.efficiency_metrics[p] for p in projects]
        
        fig.add_trace(go.Bar(
            x=projects,
            y=efficiencies,
            text=[f"{e:.1%}" for e in efficiencies],
            textposition="auto"
        ))
        
        fig.update_layout(
            title="Project Efficiency Comparison",
            xaxis_title="Project",
            yaxis_title="Efficiency Score",
            yaxis_range=[0, 1]
        )
        
        return fig
        
    def _render_resource_management(self, project_id: str, resource_type: ResourceType):
        """Render resource management interface"""
        allocations = [a for a in self.budgets[project_id] 
                      if a.resource_type == resource_type]
        
        if allocations:
            total_allocated = sum(a.amount for a in allocations)
            used = sum(u.amount_used for u in self.usage_history 
                      if u.project_id == project_id 
                      and u.resource_type == resource_type)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Allocated",
                    f"${total_allocated:,.2f}",
                    delta=f"${total_allocated - used:,.2f} remaining"
                )
                
            with col2:
                efficiency = self.efficiency_metrics.get(project_id, 0)
                st.metric(
                    "Efficiency",
                    f"{efficiency:.1%}",
                    delta=f"{(efficiency - 0.5) * 100:.1f}% vs target"
                ) 