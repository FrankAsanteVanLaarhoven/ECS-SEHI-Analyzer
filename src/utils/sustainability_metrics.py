import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

class SustainabilityMetrics:
    """Calculate sustainability and ROI metrics for green energy products."""
    
    def __init__(self):
        self.carbon_factors = {
            'traditional_catalyst': 25.5,  # kg CO2/kg
            'green_catalyst': 12.3,       # kg CO2/kg
            'fuel_cell': 8.7,            # kg CO2/kW
            'battery_material': 15.2      # kg CO2/kg
        }
        
        self.cost_factors = {
            'raw_materials': 100,         # $/kg
            'processing': 50,             # $/hour
            'energy': 0.12,               # $/kWh
            'labor': 75                   # $/hour
        }
    
    def _calculate_carbon_savings(self, data: Dict[str, float]) -> Dict[str, float]:
        """Calculate carbon savings compared to traditional methods."""
        traditional_emissions = sum(
            data.get(k, 0) * v for k, v in self.carbon_factors.items()
        )
        green_emissions = traditional_emissions * 0.6  # 40% reduction
        
        return {
            'traditional_emissions': traditional_emissions,
            'green_emissions': green_emissions,
            'savings': traditional_emissions - green_emissions
        } 

    def _calculate_costs(self, data: Dict[str, float]) -> Dict[str, float]:
        """Calculate production costs."""
        try:
            # Calculate raw material costs
            material_cost = sum(
                data.get(k, 0) * self.cost_factors['raw_materials']
                for k in ['green_catalyst', 'battery_material']
            )
            
            # Estimate processing costs (assuming 2 hours per kg)
            processing_hours = sum(data.get(k, 0) * 2 for k in ['green_catalyst', 'battery_material'])
            processing_cost = processing_hours * self.cost_factors['processing']
            
            # Estimate energy costs (assuming 5 kWh per kg)
            energy_consumption = sum(data.get(k, 0) * 5 for k in ['green_catalyst', 'battery_material'])
            energy_cost = energy_consumption * self.cost_factors['energy']
            
            # Calculate labor costs
            labor_cost = processing_hours * self.cost_factors['labor']
            
            total_cost = material_cost + processing_cost + energy_cost + labor_cost
            
            return {
                'material_cost': material_cost,
                'processing_cost': processing_cost,
                'energy_cost': energy_cost,
                'labor_cost': labor_cost,
                'total_cost': total_cost
            }
        except Exception as e:
            st.error(f"Failed to calculate costs: {str(e)}")
            return {
                'material_cost': 0,
                'processing_cost': 0,
                'energy_cost': 0,
                'labor_cost': 0,
                'total_cost': 0
            }

    def _calculate_roi(self, cost_analysis: Dict[str, float]) -> float:
        """Calculate Return on Investment."""
        try:
            # Estimate revenue (assuming 50% margin)
            revenue = cost_analysis['total_cost'] * 1.5
            
            # Calculate ROI
            roi = (revenue - cost_analysis['total_cost']) / cost_analysis['total_cost']
            
            return roi
        except Exception as e:
            st.error(f"Failed to calculate ROI: {str(e)}")
            return 0.0

    def _calculate_sustainability_score(self, carbon_savings: Dict[str, float]) -> float:
        """Calculate overall sustainability score."""
        try:
            # Base score on carbon savings percentage
            base_score = carbon_savings['savings'] / carbon_savings['traditional_emissions']
            
            # Adjust score based on absolute savings
            if carbon_savings['savings'] > 1000:  # More than 1000 kg CO2 saved
                base_score *= 1.2
            
            # Normalize score to 0-1 range
            return min(max(base_score, 0), 1)
        except Exception as e:
            st.error(f"Failed to calculate sustainability score: {str(e)}")
            return 0.0

    def calculate_metrics(self, production_data: Dict[str, float]) -> Optional[Dict[str, Any]]:
        """Calculate all sustainability and cost metrics."""
        try:
            # Calculate carbon savings
            carbon_savings = self._calculate_carbon_savings(production_data)
            
            # Calculate costs
            cost_analysis = self._calculate_costs(production_data)
            
            # Calculate ROI
            roi = self._calculate_roi(cost_analysis)
            
            # Calculate sustainability score
            sustainability_score = self._calculate_sustainability_score(carbon_savings)
            
            return {
                'carbon_savings': carbon_savings,
                'cost_analysis': cost_analysis,
                'roi': roi,
                'sustainability_score': sustainability_score
            }
        except Exception as e:
            st.error(f"Failed to calculate metrics: {str(e)}")
            return None 

class SustainabilityAnalyzer:
    """Handles sustainability metrics analysis and visualization."""
    
    def __init__(self):
        self.metric_colors = {
            'Energy Usage': '#FF4B4B',
            'Material Efficiency': '#00CC96',
            'Waste Reduction': '#AB63FA',
            'Carbon Footprint': '#FFA15A'
        }
        
        self.metric_units = {
            'Energy Usage': 'kWh',
            'Material Efficiency': '%',
            'Waste Reduction': 'kg',
            'Carbon Footprint': 'kg CO2e'
        }
    
    def calculate_sustainability_metrics(self, time_period: str, metrics: List[str]) -> None:
        """Calculate and visualize sustainability metrics."""
        
        # Generate time points based on period
        dates, data = self._generate_time_series(time_period, metrics)
        
        # Create main metrics visualization
        fig = go.Figure()
        
        for metric in metrics:
            fig.add_trace(go.Scatter(
                x=dates,
                y=data[metric],
                name=metric,
                line=dict(color=self.metric_colors[metric], width=2),
                hovertemplate=(
                    f"{metric}<br>" +
                    "Date: %{x}<br>" +
                    f"Value: %{{y:.1f}} {self.metric_units[metric]}<br>" +
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title="Sustainability Metrics Over Time",
            xaxis_title="Date",
            yaxis_title="Value",
            template="plotly_dark",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show current metrics and trends
        st.subheader("Current Metrics")
        cols = st.columns(len(metrics))
        
        for i, metric in enumerate(metrics):
            current_value = data[metric][-1]
            previous_value = data[metric][-2]
            change = ((current_value - previous_value) / previous_value) * 100
            
            with cols[i]:
                st.metric(
                    metric,
                    f"{current_value:.1f} {self.metric_units[metric]}",
                    f"{change:+.1f}%",
                    delta_color="inverse" if metric in ['Energy Usage', 'Carbon Footprint'] else "normal"
                )
        
        # Show detailed analysis
        st.subheader("Detailed Analysis")
        for metric in metrics:
            with st.expander(f"{metric} Analysis", expanded=True):
                self._show_metric_details(metric, dates, data[metric])
    
    def _generate_time_series(self, time_period: str, metrics: List[str]) -> tuple:
        """Generate time series data for metrics."""
        if time_period == "Daily":
            dates = [datetime.now() - timedelta(days=x) for x in range(30)][::-1]
            points = 30
        elif time_period == "Weekly":
            dates = [datetime.now() - timedelta(weeks=x) for x in range(12)][::-1]
            points = 12
        elif time_period == "Monthly":
            dates = [datetime.now() - timedelta(days=30*x) for x in range(12)][::-1]
            points = 12
        else:  # Yearly
            dates = [datetime.now() - timedelta(days=365*x) for x in range(5)][::-1]
            points = 5
        
        data = {}
        for metric in metrics:
            if metric in ['Energy Usage', 'Carbon Footprint']:
                # Generate decreasing trend (improvement)
                base = np.linspace(100, 70, points) + np.random.normal(0, 5, points)
            else:
                # Generate increasing trend (improvement)
                base = np.linspace(70, 90, points) + np.random.normal(0, 5, points)
            data[metric] = np.clip(base, 0, 100)
        
        return dates, data
    
    def _show_metric_details(self, metric: str, dates: List[datetime], values: np.ndarray) -> None:
        """Show detailed analysis for a specific metric."""
        col1, col2 = st.columns(2)
        
        with col1:
            # Show trend analysis
            trend = np.polyfit(range(len(values)), values, 1)[0]
            st.metric(
                "Trend",
                f"{abs(trend):.2f} {self.metric_units[metric]}/period",
                "Improving" if (trend < 0 and metric in ['Energy Usage', 'Carbon Footprint']) or
                              (trend > 0 and metric not in ['Energy Usage', 'Carbon Footprint'])
                else "Needs Attention"
            )
            
            # Show statistics
            st.write(f"Mean: {np.mean(values):.1f} {self.metric_units[metric]}")
            st.write(f"Std Dev: {np.std(values):.1f} {self.metric_units[metric]}")
        
        with col2:
            # Show distribution
            fig = go.Figure(data=[go.Histogram(
                x=values,
                nbinsx=10,
                marker_color=self.metric_colors[metric]
            )])
            
            fig.update_layout(
                title=f"{metric} Distribution",
                xaxis_title=f"Value ({self.metric_units[metric]})",
                yaxis_title="Frequency",
                template="plotly_dark",
                height=200,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True) 