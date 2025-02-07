import streamlit as st
import numpy as np
import plotly.express as px

class DefectVisualizer:
    def __init__(self):
        self.defect_colors = ['green', 'yellow', 'orange', 'red']

    def visualize_defects(self, base_data, defect_map):
        """Render defect map overlay visualization"""
        fig = px.imshow(
            base_data,
            color_continuous_scale='gray',
            title="Defect Distribution Map"
        )
        
        # Add defect overlay
        y_defects, x_defects = np.where(defect_map > 0)
        fig.add_trace(
            px.scatter(
                x=x_defects,
                y=y_defects,
                color=defect_map[defect_map > 0],
                color_continuous_scale=self.defect_colors
            ).data[0]
        )
        
        fig.update_layout(
            coloraxis_showscale=False,
            xaxis_visible=False,
            yaxis_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
