import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class SustainabilityMetrics:
    def __init__(self):
        self.metrics = {
            'energy_usage': [],
            'carbon_footprint': [],
            'resource_efficiency': [],
            'waste_reduction': []
        }
        self._initialize_sample_data()
        
    def _initialize_sample_data(self):
        """Initialize sample metrics data"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate sample data for each metric
        for date in dates:
            self.track_metrics(
                energy_usage=np.random.normal(100, 10),
                carbon_footprint=np.random.normal(50, 5),
                resource_efficiency=np.random.normal(85, 5),
                waste_reduction=np.random.normal(75, 8)
            )
    
    def get_current_metric(self, metric_name: str) -> float:
        """Get current value for a metric"""
        if not self.metrics[metric_name]:
            return 0.0
        return self.metrics[metric_name][-1]['value']
    
    def get_metric_trend(self, metric_name: str, days: int = 7) -> float:
        """Calculate trend for a metric"""
        if len(self.metrics[metric_name]) < 2:
            return 0.0
            
        recent = self.metrics[metric_name][-1]['value']
        previous = self.metrics[metric_name][-days]['value']
        return ((recent - previous) / previous) * 100
    
    def track_metrics(self, **kwargs):
        """Track sustainability metrics"""
        timestamp = datetime.now()
        for metric, value in kwargs.items():
            if metric in self.metrics:
                self.metrics[metric].append({
                    'value': value,
                    'timestamp': timestamp
                })
                
    def plot_efficiency_trends(self):
        """Plot efficiency trends"""
        fig = go.Figure()
        
        for metric_name, data in self.metrics.items():
            if data:
                dates = [d['timestamp'] for d in data]
                values = [d['value'] for d in data]
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=values,
                    name=metric_name.replace('_', ' ').title(),
                    mode='lines+markers'
                ))
                
        fig.update_layout(
            title="Sustainability Metrics Over Time",
            xaxis_title="Date",
            yaxis_title="Value",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    def render_sustainability_dashboard(self):
        """Render sustainability metrics dashboard"""
        st.markdown("### 🌱 Sustainability Metrics")
        
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            energy = self.get_current_metric('energy_usage')
            energy_trend = self.get_metric_trend('energy_usage')
            st.metric(
                "Energy Usage",
                f"{energy:.1f} kWh",
                f"{energy_trend:+.1f}%"
            )
            
        with col2:
            carbon = self.get_current_metric('carbon_footprint')
            carbon_trend = self.get_metric_trend('carbon_footprint')
            st.metric(
                "Carbon Footprint",
                f"{carbon:.1f} kg CO2e",
                f"{carbon_trend:+.1f}%"
            )
            
        with col3:
            efficiency = self.get_current_metric('resource_efficiency')
            efficiency_trend = self.get_metric_trend('resource_efficiency')
            st.metric(
                "Resource Efficiency",
                f"{efficiency:.1f}%",
                f"{efficiency_trend:+.1f}%"
            )
            
        with col4:
            waste = self.get_current_metric('waste_reduction')
            waste_trend = self.get_metric_trend('waste_reduction')
            st.metric(
                "Waste Reduction",
                f"{waste:.1f}%",
                f"{waste_trend:+.1f}%"
            )
            
        # Efficiency trends
        st.markdown("#### Efficiency Trends")
        self.plot_efficiency_trends()
        
        # Settings and controls
        with st.expander("⚙️ Sustainability Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.selectbox(
                    "Metric Display",
                    ["All", "Energy Only", "Carbon Only", "Resources Only"],
                    key="sustainability_display"
                )
                st.slider(
                    "Update Interval (min)",
                    1, 60, 5,
                    key="sustainability_interval"
                )
                
            with col2:
                st.checkbox("Enable Alerts", key="sustainability_alerts")
                st.number_input(
                    "Alert Threshold",
                    0.0, 100.0, 90.0,
                    key="sustainability_threshold"
                ) 