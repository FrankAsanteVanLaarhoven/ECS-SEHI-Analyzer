import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

def render_sustainability_metrics():
    """Render the sustainability metrics page."""
    st.markdown('<h1 class="main-header">Sustainability Metrics</h1>', unsafe_allow_html=True)
    
    # Create layout
    left_col, main_col = st.columns([1, 3])
    
    with left_col:
        st.markdown('<div class="control-panel">', unsafe_allow_html=True)
        st.subheader("Sustainability Controls")
        
        # Time period selection
        time_period = st.selectbox(
            "Time Period",
            ["Last Week", "Last Month", "Last Quarter", "Last Year"],
            help="Select time period for analysis"
        )
        
        # Metric selection
        metrics = st.multiselect(
            "Metrics",
            [
                "Energy Usage",
                "Water Consumption",
                "Material Efficiency",
                "Waste Generation",
                "Carbon Footprint",
                "Resource Recovery",
                "Process Optimization"
            ],
            default=["Energy Usage", "Water Consumption"],
            help="Select metrics to analyze"
        )
        
        # Generate Report button
        if st.button("Generate Report", type="primary"):
            with main_col:
                # Generate sample data
                dates = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
                data = {}
                
                for metric in metrics:
                    # Generate realistic-looking data with trends
                    if metric in ["Energy Usage", "Water Consumption"]:
                        base = np.random.normal(50, 10, len(dates))
                        trend = np.linspace(0, 5, len(dates))
                        data[metric] = base + trend
                    else:
                        data[metric] = np.random.normal(75, 15, len(dates))
                
                df = pd.DataFrame(data, index=dates)
                
                # Display summary metrics
                st.subheader("Summary Metrics")
                cols = st.columns(len(metrics))
                for col, metric in zip(cols, metrics):
                    current = df[metric].iloc[-1]
                    previous = df[metric].iloc[-2]
                    delta = ((current - previous) / previous) * 100
                    col.metric(
                        metric,
                        f"{current:.1f}",
                        f"{delta:+.1f}%"
                    )
                
                # Show trend visualization
                st.subheader("Trends Over Time")
                fig = go.Figure()
                
                for metric in metrics:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df[metric],
                        name=metric,
                        mode='lines+markers'
                    ))
                
                fig.update_layout(
                    title="Sustainability Metrics Trends",
                    xaxis_title="Date",
                    yaxis_title="Value",
                    template="plotly_dark",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show distribution analysis
                st.subheader("Distribution Analysis")
                fig = go.Figure()
                
                for metric in metrics:
                    fig.add_trace(go.Box(
                        y=df[metric],
                        name=metric,
                        boxpoints='all',
                        jitter=0.3,
                        pointpos=-1.8
                    ))
                
                fig.update_layout(
                    title="Metric Distributions",
                    yaxis_title="Value",
                    template="plotly_dark",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show correlation heatmap if multiple metrics
                if len(metrics) > 1:
                    st.subheader("Metric Correlations")
                    corr = df[metrics].corr()
                    
                    fig = px.imshow(
                        corr,
                        labels=dict(color="Correlation"),
                        x=metrics,
                        y=metrics,
                        color_continuous_scale="RdBu"
                    )
                    
                    fig.update_layout(
                        title="Correlation Heatmap",
                        template="plotly_dark",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Add sustainability recommendations
                st.subheader("Recommendations")
                for metric in metrics:
                    with st.expander(f"{metric} Optimization"):
                        current_value = df[metric].iloc[-1]
                        target = current_value * 0.9  # 10% reduction target
                        
                        st.write(f"Current {metric}: {current_value:.1f}")
                        st.write(f"Target {metric}: {target:.1f}")
                        st.write("Recommendations:")
                        st.write("• Implement automated monitoring systems")
                        st.write("• Optimize process parameters")
                        st.write("• Regular maintenance and calibration")
                        st.write("• Staff training on best practices")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main visualization area
    with main_col:
        if "sustainability_metrics_btn" not in st.session_state:
            st.markdown('<div class="visualization-area">', unsafe_allow_html=True)
            st.markdown("""
                <div style="text-align: center; padding: 40px;">
                    <h3 style="color: #94A3B8;">Sustainability Metrics</h3>
                    <p style="color: #64748B;">
                        Select metrics and parameters, then click 'Generate Metrics' 
                        to view sustainability analysis results.
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True) 