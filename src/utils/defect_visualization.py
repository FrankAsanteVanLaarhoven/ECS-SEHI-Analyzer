import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, List, Tuple
import streamlit as st
import plotly.express as px
import pandas as pd
from scipy import ndimage

class DefectVisualizer:
    """Enhanced visualization for defect analysis."""
    
    def __init__(self):
        self.defect_colors = {
            "Cracks": "#FF4B4B",
            "Voids": "#4169E1",
            "Delamination": "#FFA500",
            "Inclusions": "#32CD32",
            "Porosity": "#9370DB",
            "Surface Contamination": "#FF69B4",
            "Grain Boundaries": "#20B2AA",
            "Phase Separation": "#FFD700"
        }
        
        self.defect_descriptions = {
            "Cracks": "Linear discontinuities in material",
            "Voids": "Empty spaces or cavities",
            "Delamination": "Layer separation in composites",
            "Inclusions": "Foreign material embedment",
            "Porosity": "Network of small voids",
            "Surface Contamination": "Unwanted surface deposits",
            "Grain Boundaries": "Crystal structure interfaces",
            "Phase Separation": "Material composition variations"
        }

    def create_defect_plot(self, data, confidence_map=None, defect_types=None):
        """Create enhanced defect visualization."""
        try:
            # Extract surface data from dictionary if needed
            if isinstance(data, dict):
                surface_data = data['surface']
                X = data['X']
                Y = data['Y']
                defect_mask = data['defects']
            else:
                surface_data = data
                X, Y = np.meshgrid(np.linspace(-1, 1, data.shape[0]), 
                                 np.linspace(-1, 1, data.shape[1]))
                defect_mask = None

            # Create figure
            fig = go.Figure()

            # Add surface plot
            fig.add_trace(go.Surface(
                x=X,
                y=Y,
                z=surface_data,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text="Height",
                        side="right"
                    ),
                    x=1.1,
                    len=0.75
                ),
                lighting=dict(
                    ambient=0.8,
                    diffuse=0.9,
                    fresnel=0.2,
                    specular=1,
                    roughness=0.5
                ),
                contours=dict(
                    x=dict(show=True, width=2),
                    y=dict(show=True, width=2),
                    z=dict(show=True, width=2)
                )
            ))

            # Add defect markers if available
            if defect_mask is not None:
                defect_positions = np.where(defect_mask)
                if len(defect_positions[0]) > 0:
                    fig.add_trace(go.Scatter3d(
                        x=X[defect_positions],
                        y=Y[defect_positions],
                        z=surface_data[defect_positions],
                        mode='markers',
                        marker=dict(
                            size=5,
                            color='red',
                            symbol='circle',
                            line=dict(color='rgba(255,0,0,0.8)', width=2)
                        ),
                        name='Defects'
                    ))

            # Update layout
            fig.update_layout(
                title={
                    'text': "3D Surface Analysis",
                    'y': 0.95,
                    'x': 0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'
                },
                scene=dict(
                    xaxis_title="X Position (μm)",
                    yaxis_title="Y Position (μm)",
                    zaxis_title="Height (nm)",
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=2)
                    ),
                    aspectmode='data'
                ),
                width=800,
                height=600,
                template="plotly_dark",
                showlegend=True,
                margin=dict(l=0, r=100, t=30, b=0)
            )

            # Add view control buttons
            fig.update_layout(
                updatemenus=[{
                    'type': 'buttons',
                    'showactive': True,
                    'buttons': [
                        dict(
                            args=[{'visible': [True, True]}],
                            label="Both",
                            method="restyle"
                        ),
                        dict(
                            args=[{'visible': [True, False]}],
                            label="Surface Only",
                            method="restyle"
                        ),
                        dict(
                            args=[{'visible': [False, True]}],
                            label="Defects Only",
                            method="restyle"
                        )
                    ],
                    'x': 0.1,
                    'y': 1.1,
                    'xanchor': 'left',
                    'yanchor': 'top'
                }]
            )

            return fig

        except Exception as e:
            raise Exception(f"Error creating defect plot: {str(e)}")

    def show_analysis_tools(self, defect_map: np.ndarray, confidence_map: np.ndarray):
        """Display interactive analysis tools."""
        st.subheader("Analysis Tools")
        
        # Create tabs for different analysis tools
        tabs = st.tabs(["Profile Analysis", "Region Statistics", "Distribution Analysis"])
        
        with tabs[0]:
            self._show_profile_analysis(defect_map, confidence_map)
            
        with tabs[1]:
            self._show_region_statistics(defect_map, confidence_map)
            
        with tabs[2]:
            self._show_distribution_analysis(defect_map)

    def _show_profile_analysis(self, defect_map: np.ndarray, confidence_map: np.ndarray):
        """Show line profile analysis tool."""
        col1, col2 = st.columns([1, 2])
        
        with col1:
            profile_type = st.selectbox(
                "Profile Type",
                ["Horizontal", "Vertical", "Custom"]
            )
            
            if profile_type in ["Horizontal", "Vertical"]:
                position = st.slider(
                    "Position",
                    0, defect_map.shape[0]-1,
                    defect_map.shape[0]//2
                )
        
        with col2:
            fig = go.Figure()
            
            if profile_type == "Horizontal":
                profile = defect_map[position, :]
                confidence = confidence_map[position, :]
                x = np.arange(len(profile))
                xlabel = "X Position"
            else:  # Vertical
                profile = defect_map[:, position]
                confidence = confidence_map[:, position]
                x = np.arange(len(profile))
                xlabel = "Y Position"
                
            fig.add_trace(go.Scatter(
                x=x, y=profile,
                name="Defect Intensity",
                line=dict(color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=x, y=confidence,
                name="Confidence",
                line=dict(color='blue')
            ))
            
            fig.update_layout(
                title="Profile Analysis",
                xaxis_title=xlabel,
                yaxis_title="Intensity",
                template="plotly_dark"
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def _show_region_statistics(self, defect_map: np.ndarray, confidence_map: np.ndarray):
        """Show regional statistics analysis."""
        # Calculate regional statistics
        labeled, num = ndimage.label(defect_map > 0.5)
        
        if num > 0:
            regions = []
            for i in range(1, num+1):
                region_mask = labeled == i
                region = {
                    'ID': i,
                    'Size': np.sum(region_mask),
                    'Mean Intensity': np.mean(defect_map[region_mask]),
                    'Max Intensity': np.max(defect_map[region_mask]),
                    'Mean Confidence': np.mean(confidence_map[region_mask])
                }
                regions.append(region)
            
            df = pd.DataFrame(regions)
            st.dataframe(df)
            
            # Show histogram of region sizes
            fig = px.histogram(
                df, x='Size',
                title="Distribution of Defect Sizes",
                template="plotly_dark"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No distinct regions detected")

    def _show_distribution_analysis(self, defect_map: np.ndarray):
        """Show defect distribution analysis."""
        # Calculate intensity distribution
        hist, bins = np.histogram(defect_map[defect_map > 0], bins=50)
        
        fig = go.Figure(data=[
            go.Bar(
                x=bins[:-1],
                y=hist,
                name="Intensity Distribution"
            )
        ])
        
        fig.update_layout(
            title="Defect Intensity Distribution",
            xaxis_title="Intensity",
            yaxis_title="Count",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def create_distribution_plot(self, data):
        """Create visualization of height and defect distributions."""
        try:
            if isinstance(data, dict) and 'surface' in data:
                surface_data = data['surface']
                defect_mask = data.get('defects', None)
                
                # Create figure with two subplots
                fig = go.Figure()
                
                # Height distribution
                heights = surface_data.flatten()
                fig.add_trace(go.Histogram(
                    x=heights,
                    name='Height Distribution',
                    nbinsx=50,
                    marker_color='#00ff00',
                    opacity=0.7
                ))
                
                # Defect distribution if available
                if defect_mask is not None:
                    defect_heights = surface_data[defect_mask]
                    if len(defect_heights) > 0:
                        fig.add_trace(go.Histogram(
                            x=defect_heights,
                            name='Defect Distribution',
                            nbinsx=50,
                            marker_color='#ff0000',
                            opacity=0.7
                        ))
                
                # Update layout
                fig.update_layout(
                    title="Height and Defect Distribution",
                    xaxis_title="Height (nm)",
                    yaxis_title="Count",
                    template="plotly_dark",
                    height=400,
                    barmode='overlay',
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01,
                        bgcolor="rgba(0,0,0,0.5)"
                    ),
                    margin=dict(l=50, r=50, t=50, b=50)
                )
                
                # Add statistical annotations
                fig.add_annotation(
                    x=np.mean(heights),
                    y=0,
                    text=f"Mean: {np.mean(heights):.2f} nm",
                    showarrow=True,
                    arrowhead=1,
                    yshift=20
                )
                
                fig.add_annotation(
                    x=np.median(heights),
                    y=0,
                    text=f"Median: {np.median(heights):.2f} nm",
                    showarrow=True,
                    arrowhead=1,
                    yshift=40
                )
                
                # Add buttons to toggle between distributions
                fig.update_layout(
                    updatemenus=[
                        dict(
                            type="buttons",
                            direction="right",
                            x=0.1,
                            y=1.1,
                            showactive=True,
                            buttons=[
                                dict(
                                    label="Both",
                                    method="update",
                                    args=[{"visible": [True, True]}]
                                ),
                                dict(
                                    label="Heights Only",
                                    method="update",
                                    args=[{"visible": [True, False]}]
                                ),
                                dict(
                                    label="Defects Only",
                                    method="update",
                                    args=[{"visible": [False, True]}]
                                )
                            ]
                        )
                    ]
                )
                
                return fig
            else:
                raise ValueError("Invalid data format. Expected dictionary with 'surface' key.")
            
        except Exception as e:
            raise Exception(f"Error creating distribution plot: {str(e)}")
        
    def _calculate_defect_stats(self, defect_map: np.ndarray) -> Dict[int, int]:
        """Calculate statistics for each defect type."""
        try:
            stats = {}
            for i in range(1, len(self.defect_types) + 1):
                count = np.sum(defect_map == i)
                stats[i] = int(count)  # Convert to int for safe JSON serialization
            return stats
        except Exception as e:
            st.error(f"Failed to calculate defect statistics: {str(e)}")
            return {i: 0 for i in range(1, len(self.defect_types) + 1)}

    def _calculate_quality_score(self, defect_stats: Dict[int, int], total_area: int) -> float:
        """Calculate overall quality score based on defect statistics."""
        try:
            if total_area == 0:
                return 1.0

            score = 1.0
            for defect_id, count in defect_stats.items():
                defect_type = list(self.defect_types.keys())[defect_id - 1]
                defect_info = self.defect_types[defect_type]
                defect_percentage = count / total_area
                score -= defect_percentage * defect_info['weight']

            return max(min(score, 1.0), 0.0)
        except Exception as e:
            st.error(f"Failed to calculate quality score: {str(e)}")
            return 0.0

    def visualize_defects(self, data: np.ndarray, defect_map: np.ndarray) -> None:
        """Create interactive 3D defect visualization."""
        try:
            # Create coordinate meshgrid
            y, x = np.mgrid[0:data.shape[0], 0:data.shape[1]]
            
            # Create main figure
            fig = go.Figure()

            # Add surface plot
            fig.add_trace(go.Surface(
                x=x,
                y=y,
                z=data,
                colorscale='Viridis',
                opacity=0.8,
                name='Surface',
                showscale=True,
                colorbar=dict(
                    title=dict(
                        text='Height (nm)',
                        side='right'
                    ),
                    x=1.1
                )
            ))

            # Add defect markers for each type
            for defect_id, defect_type in enumerate(self.defect_types.keys(), start=1):
                defect_info = self.defect_types[defect_type]
                mask = defect_map == defect_id
                
                if np.any(mask):
                    # Get coordinates where defects are present
                    y_coords, x_coords = np.where(mask)
                    z_coords = data[mask]

                    fig.add_trace(go.Scatter3d(
                        x=x_coords,
                        y=y_coords,
                        z=z_coords,
                        mode='markers',
                        name=f"{defect_info['symbol']} {defect_type.title()}",
                        marker=dict(
                            size=5,
                            color=defect_info['color'],
                            symbol='circle',
                            line=dict(color='white', width=1)
                        ),
                        hovertemplate=(
                            f"<b>{defect_type.title()}</b><br>" +
                            "X: %{x}<br>" +
                            "Y: %{y}<br>" +
                            "Z: %{z:.2f} nm<br>" +
                            f"Severity: {defect_info['severity'].upper()}<br>" +
                            "<extra></extra>"
                        )
                    ))

            # Update layout
            fig.update_layout(
                title=dict(
                    text="3D Defect Analysis",
                    x=0.5,
                    y=0.95
                ),
                scene=dict(
                    xaxis_title="X Position (μm)",
                    yaxis_title="Y Position (μm)",
                    zaxis_title="Height (nm)",
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=1.2)
                    ),
                    aspectmode='data'
                ),
                template="plotly_dark",
                showlegend=True,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(0,0,0,0.5)"
                ),
                height=700
            )

            st.plotly_chart(fig, use_container_width=True)

            # Calculate and display statistics
            stats = self._calculate_defect_stats(defect_map)
            quality_score = self._calculate_quality_score(stats, defect_map.size)

            # Display metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Quality Score",
                    f"{quality_score:.1%}",
                    delta=self._get_quality_description(quality_score),
                    delta_color="inverse"
                )

            with col2:
                total_defects = sum(stats.values())
                st.metric(
                    "Total Defects",
                    f"{total_defects:,}",
                    f"{(total_defects/defect_map.size):.1%} of area",
                    delta_color="inverse"
                )

            # Display defect breakdown
            st.subheader("Defect Analysis")
            for defect_id, count in stats.items():
                defect_type = list(self.defect_types.keys())[defect_id - 1]
                defect_info = self.defect_types[defect_type]
                st.markdown(
                    f"{defect_info['symbol']} **{defect_type.title()}**: "
                    f"{count:,} instances ({(count/defect_map.size)*100:.1f}% of area) - "
                    f"Severity: {defect_info['severity'].upper()}"
                )

        except Exception as e:
            st.error(f"Failed to visualize defects: {str(e)}")
            st.exception(e)

    def _get_quality_description(self, score: float) -> str:
        """Get descriptive text for quality score."""
        if score >= 0.9:
            return "Excellent quality"
        elif score >= 0.8:
            return "Good quality"
        elif score >= 0.6:
            return "Fair quality"
        else:
            return "Needs improvement" 

    def create_summary_metrics(self, stats: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Create summary metrics for dashboard display."""
        try:
            return {
                "Total Defects": {
                    "value": stats['total_defects'],
                    "format": "{:,}",
                    "delta": None
                },
                "Average Size": {
                    "value": stats['avg_size'],
                    "format": "{:.2f} nm",
                    "delta": None
                },
                "Coverage": {
                    "value": stats['coverage'],
                    "format": "{:.2%}",
                    "delta": None
                },
                "Confidence": {
                    "value": stats['confidence'],
                    "format": "{:.2%}",
                    "delta": None
                }
            }
        except Exception as e:
            raise Exception(f"Error creating summary metrics: {str(e)}")

    def create_component_plot(self, components: np.ndarray, algorithm: str) -> go.Figure:
        """Create visualization for algorithm components."""
        fig = go.Figure()
        
        for i in range(components.shape[0]):
            fig.add_trace(go.Surface(
                z=components[i],
                name=f"Component {i+1}",
                colorscale='Viridis'
            ))
        
        fig.update_layout(
            title=f"{algorithm} Components",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Intensity"
            ),
            template="plotly_dark"
        )
        
        return fig

    def create_network_plot(self, network_metrics: Dict[str, float]) -> go.Figure:
        """Create visualization for carbon network properties."""
        # Create radar chart for network properties
        fig = go.Figure()
        
        # Prepare data for radar chart
        categories = ['sp2/sp3 Ratio', 'Crystallinity', 'Surface Area', 'Porosity']
        values = [
            network_metrics['sp2_sp3_ratio'],
            network_metrics['crystallinity'],
            network_metrics['surface_area'] / 1000,  # Normalize surface area
            network_metrics['porosity']
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Network Properties'
        ))
        
        fig.update_layout(
            title="Carbon Network Properties",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False,
            template="plotly_dark"
        )
        
        return fig

    def create_functional_groups_plot(self, functional_groups: Dict[str, float]) -> go.Figure:
        """Create visualization for functional groups distribution."""
        fig = go.Figure()
        
        # Create bar chart for functional groups
        fig.add_trace(go.Bar(
            x=list(functional_groups.keys()),
            y=list(functional_groups.values()),
            marker_color='rgb(55, 83, 109)',
            text=[f"{v:.1%}" for v in functional_groups.values()],
            textposition='auto',
        ))
        
        fig.update_layout(
            title="Functional Groups Distribution",
            xaxis_title="Functional Group",
            yaxis_title="Relative Abundance",
            template="plotly_dark",
            showlegend=False,
            xaxis_tickangle=-45
        )
        
        return fig

    def create_degradation_plot(self, degradation_metrics: Dict[str, float]) -> go.Figure:
        """Create visualization for degradation assessment."""
        fig = go.Figure()
        
        # Create gauge charts for each metric
        fig = go.Figure()
        
        # Add gauge charts for each metric
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=degradation_metrics['mechanical_stability'],
            title={'text': "Mechanical Stability"},
            domain={'x': [0, 0.3], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "lightblue"},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "green"}
                ]
            }
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=degradation_metrics['chemical_stability'] * 100,
            title={'text': "Chemical Stability"},
            domain={'x': [0.35, 0.65], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "lightblue"},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 75], 'color': "yellow"},
                    {'range': [75, 100], 'color': "green"}
                ]
            }
        ))
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=degradation_metrics['corrosion_rate'],
            title={'text': "Corrosion Rate (mm/year)"},
            domain={'x': [0.7, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 1]},
                'bar': {'color': "lightblue"},
                'steps': [
                    {'range': [0.5, 1], 'color': "red"},
                    {'range': [0.2, 0.5], 'color': "yellow"},
                    {'range': [0, 0.2], 'color': "green"}
                ]
            }
        ))
        
        fig.update_layout(
            title="Degradation Assessment",
            template="plotly_dark",
            height=400
        )
        
        return fig

    def create_loading_plot(self, loadings: np.ndarray) -> go.Figure:
        """Create visualization for component loadings."""
        fig = go.Figure()
        
        for i in range(loadings.shape[1]):
            fig.add_trace(go.Scatter(
                y=loadings[:, i],
                name=f"Component {i+1}",
                mode='lines'
            ))
        
        fig.update_layout(
            title="Component Loadings",
            xaxis_title="Energy (eV)",
            yaxis_title="Loading",
            template="plotly_dark",
            showlegend=True
        )
        
        return fig

    def create_time_series_plot(self, time_series_data: Dict[str, np.ndarray]) -> go.Figure:
        """Create visualization for time series analysis."""
        fig = go.Figure()
        
        for metric, values in time_series_data.items():
            fig.add_trace(go.Scatter(
                y=values,
                name=metric,
                mode='lines+markers'
            ))
        
        fig.update_layout(
            title="Analysis Results Over Time",
            xaxis_title="Time",
            yaxis_title="Value",
            template="plotly_dark",
            showlegend=True
        )
        
        return fig

    def generate_sample_data(self, resolution):
        """Generate sample defect data for visualization."""
        x = np.linspace(-1, 1, resolution)
        y = np.linspace(-1, 1, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Create base surface
        R = np.sqrt(X**2 + Y**2)
        Z = (1 - R) * 2
        
        return {
            'surface': Z,
            'X': X,
            'Y': Y,
            'defects': np.random.rand(resolution, resolution) < 0.1
        }

    def create_cross_section_plot(self, data):
        """Create cross-section analysis plot."""
        fig = go.Figure()
        
        # Add X cross-section
        mid_point = data['surface'].shape[0] // 2
        fig.add_trace(go.Scatter(
            y=data['surface'][mid_point, :],
            name='X Cross-section'
        ))
        
        fig.update_layout(
            title="Cross-section Analysis",
            template="plotly_dark",
            height=400
        )
        
        return fig

    def create_evolution_plot(self, data):
        """Create time evolution plot."""
        fig = go.Figure()
        
        # Create time series data
        time_points = 10
        frames = []
        
        for t in range(time_points):
            evolved_surface = data['surface'] + np.random.randn(*data['surface'].shape) * 0.05 * t
            frames.append(
                go.Frame(
                    data=[go.Surface(z=evolved_surface, colorscale='Viridis')],
                    name=f'frame{t}'
                )
            )
        
        fig.frames = frames
        fig.add_trace(go.Surface(
            z=data['surface'],
            colorscale='Viridis',
            showscale=True
        ))
        
        # Add animation controls
        fig.update_layout(
            title="Defect Evolution",
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [{
                    'label': 'Play',
                    'method': 'animate',
                    'args': [None, {'frame': {'duration': 500}}]
                }]
            }],
            template="plotly_dark",
            height=600
        )
        
        return fig

    def create_nanoscale_visualization(self, sehi_data, ecs_data):
        """Create visualization for nanoscale analysis."""
        
        # Create tabs for different analysis views
        tabs = st.tabs([
            "Chemical Mapping",
            "Electronic Structure",
            "Surface Features",
            "Layer Analysis"
        ])
        
        with tabs[0]:
            # Chemical composition map
            st.subheader("Chemical Composition at Nanoscale")
            chemical_fig = self._create_chemical_map(sehi_data['chemical_composition'])
            st.plotly_chart(chemical_fig)
            
            # Show bonding states
            col1, col2 = st.columns(2)
            with col1:
                st.metric("sp² Bonding", f"{sehi_data['chemical_composition']['bonding_states']['sp2']:.1f}%")
            with col2:
                st.metric("sp³ Bonding", f"{sehi_data['chemical_composition']['bonding_states']['sp3']:.1f}%")
        
        with tabs[1]:
            # Electronic structure visualization
            st.subheader("Electronic Structure Analysis")
            electronic_fig = self._create_electronic_structure_plot(sehi_data['electronic_structure'])
            st.plotly_chart(electronic_fig)
            
            # Show key electronic properties
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Work Function", f"{sehi_data['electronic_structure']['work_function']:.1f} eV")
            with col2:
                st.metric("Band Gap", f"{sehi_data['electronic_structure']['band_gap']:.1f} eV")
        
        with tabs[2]:
            # Surface features visualization
            st.subheader("Surface Features (nm scale)")
            surface_fig = self._create_surface_features_plot(sehi_data['surface_features'])
            st.plotly_chart(surface_fig)
            
            # Show surface metrics
            metrics = sehi_data['surface_features']['topography']
            cols = st.columns(4)
            cols[0].metric("Mean Roughness", f"{metrics['mean_roughness']:.1f} nm")
            cols[1].metric("Peak Height", f"{metrics['peak_height']:.1f} nm")
            cols[2].metric("Valley Depth", f"{metrics['valley_depth']:.1f} nm")
            cols[3].metric("Feature Density", f"{metrics['feature_density']:.2f}/nm²")

        with tabs[3]:
            # Layer analysis
            st.subheader("Layer Analysis")
            layer_analysis_fig = self._create_layer_analysis_plot(ecs_data)
            st.plotly_chart(layer_analysis_fig)

    def _create_chemical_map(self, chemical_composition):
        """Create visualization for chemical composition."""
        # Implementation of _create_chemical_map method
        pass

    def _create_electronic_structure_plot(self, electronic_structure):
        """Create visualization for electronic structure."""
        # Implementation of _create_electronic_structure_plot method
        pass

    def _create_surface_features_plot(self, surface_features):
        """Create visualization for surface features."""
        # Implementation of _create_surface_features_plot method
        pass

    def _create_layer_analysis_plot(self, ecs_data):
        """Create visualization for layer analysis."""
        # Implementation of _create_layer_analysis_plot method
        pass 