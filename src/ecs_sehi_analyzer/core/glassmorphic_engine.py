# File: src/ecs_sehi_analyzer/core/glassmorphic_engine.py
import streamlit as st
import numpy as np
import plotly.graph_objects as go
from typing import Tuple
from pathlib import Path
import json

class GlassmorphicEngine:
    def __init__(self):
        self.CONFIG = {
            "glass_effect": {
                "background": "rgba(255, 255, 255, 0.15)",
                "backdrop_filter": "blur(12px)",
                "border": "1px solid rgba(255, 255, 255, 0.2)",
                "border_radius": "24px",
                "box_shadow": "0 8px 32px 0 rgba(31, 38, 135, 0.37)"
            },
            "motion": {
                "hover_scale": 1.03,
                "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)"
            }
        }

    def create_3d_globe(self) -> go.Figure:
        """Create interactive 3D globe with particle effects"""
        fig = go.Figure(go.Scattergeo())
        fig.update_geos(
            projection_type="orthographic",
            landcolor="rgba(173, 216, 230, 0.4)",
            showocean=True,
            oceancolor="rgba(70, 130, 180, 0.4)"
        )
        fig.add_trace(go.Scattergeo(
            mode="markers",
            marker=dict(
                size=8,
                color="#4B90E2",
                line=dict(width=2, color="#FFFFFF"),
                symbol="circle"
            )
        ))
        return fig

    def glassmorphic_card(self, content: str) -> str:
        """Generate CSS for glassmorphic cards with depth perception"""
        return f"""
        <style>
            .glass-card {{
                background: {self.CONFIG['glass_effect']['background']};
                backdrop-filter: {self.CONFIG['glass_effect']['backdrop_filter']};
                border: {self.CONFIG['glass_effect']['border']};
                border-radius: {self.CONFIG['glass_effect']['border_radius']};
                box-shadow: {self.CONFIG['glass_effect']['box_shadow']};
                transition: {self.CONFIG['motion']['transition']};
            }}
            .glass-card:hover {{
                transform: scale({self.CONFIG['motion']['hover_scale']});
                backdrop-filter: blur(16px);
            }}
        </style>
        <div class="glass-card">
            {content}
        </div>
        """

class InteractionDesign:
    def __init__(self):
        self.ANIMATION_CONFIG = {
            "card_magnification": 1.15,
            "transition_duration": 0.4,
            "easing": "cubic-bezier(0.68, -0.55, 0.27, 1.55)"
        }

    def parallax_hover(self, element_id: str) -> str:
        """Create 3D parallax hover effect using CSS transforms"""
        return f"""
        <script>
            document.getElementById('{element_id}').addEventListener('mousemove', (e) => {{
                const card = e.currentTarget;
                const xAxis = (window.innerWidth / 2 - e.pageX) / 25;
                const yAxis = (window.innerHeight / 2 - e.pageY) / 25;
                card.style.transform = `rotateY(${xAxis}deg) rotateX(${yAxis}deg)`;
            }});
        </script>
        """

    def infinite_scroll_loader(self) -> str:
        """Animated SVG loader for infinite scroll"""
        return """
        <div class="loader">
            <svg viewBox="0 0 50 50">
                <circle cx="25" cy="25" r="20" fill="none" stroke="#4B90E2" stroke-width="4">
                    <animate attributeName="r" values="20;5;20" dur="1.5s" repeatCount="indefinite"/>
                    <animate attributeName="opacity" values="1;0;1" dur="1.5s" repeatCount="indefinite"/>
                </circle>
            </svg>
        </div>
        """

# Implementation Example
if __name__ == "__main__":
    ge = GlassmorphicEngine()
    st.markdown(ge.glassmorphic_card("""
        <h3 style='color: white;'>Global Analytics</h3>
        <p style='color: rgba(255,255,255,0.8);'>Real-time metrics</p>
    """), unsafe_allow_html=True)
    
    fig = ge.create_3d_globe()
    st.plotly_chart(fig, use_container_width=True)
