import streamlit as st
import plotly.graph_objects as go
import numpy as np
import cv2
import logging
from utils.photogrammetry import PhotogrammetryProcessor, PhotogrammetryParameters
from utils.visualization_3d import create_3d_animation
from utils.data_manager import DataManager

def render_multi_modal_analysis():
    """Render the multi-modal analysis page."""
    st.markdown('<h1>Multi-Modal Analysis</h1>', unsafe_allow_html=True)
    
    # Left column for controls
    left_col, right_col = st.columns([1, 2])
    
    with left_col:
        st.markdown("## Multi-Modal Controls")
        
        # Analysis Modes with info icon
        st.markdown("""
            <div style="display: flex; align-items: center;">
                <span>Analysis Modes</span>
                <span class="info-icon">ⓘ</span>
            </div>
        """, unsafe_allow_html=True)
        
        # Selected modes as removable tags
        selected_modes = []
        mode_tags = st.container()
        with mode_tags:
            if st.button("Chemical-Surface ×", key="cs"):
                selected_modes.append("Chemical-Surface")
            if st.button("Surface-Defect ×", key="sd"):
                selected_modes.append("Surface-Defect")
            if st.button("Chemical-Defect ×", key="cd"):
                selected_modes.append("Chemical-Defect")
            if st.button("Photogrammetry ×", key="pg"):
                selected_modes.append("Photogrammetry")
        
        # Add dropdown for adding more modes
        available_modes = [
            mode for mode in [
                "Chemical-Surface",
                "Surface-Defect",
                "Chemical-Defect",
                "Photogrammetry"
            ] if mode not in selected_modes
        ]
        
        # Dropdown arrow button
        if st.button("▼"):
            mode_to_add = st.selectbox(
                "",
                options=available_modes,
                key="mode_selector"
            )
            if mode_to_add:
                selected_modes.append(mode_to_add)
        
        # Resolution with info icon
        st.markdown("""
            <div style="display: flex; align-items: center;">
                <span>Resolution</span>
                <span class="info-icon">ⓘ</span>
            </div>
        """, unsafe_allow_html=True)
        resolution = st.slider("", 128, 1024, 512, key="resolution")
        
        # Advanced Options
        with st.expander("Advanced Options", expanded=True):
            st.markdown("Fusion Method")
            fusion_method = st.selectbox(
                "",
                options=["Early Fusion", "Late Fusion", "Hybrid Fusion"],
                index=0
            )
            
            st.markdown("Correlation Threshold")
            correlation_threshold = st.slider("", 0.00, 1.00, 0.70)
        
        # Run Analysis Button
        st.button("Run Multi-Modal Analysis", type="primary")
        
        # Add data management section
        with st.expander("Data Management", expanded=False):
            data_manager = DataManager()
            
            # Data Upload
            st.subheader("Data Upload")
            upload_type = st.selectbox(
                "Select Data Type",
                ["SEHI", "ECS", "LiDAR", "Photogrammetry", "General"]
            )
            
            # Show supported formats
            st.caption(f"Supported formats: {', '.join(DataManager.SUPPORTED_FORMATS[upload_type])}")
            
            # File uploader
            uploaded_file = st.file_uploader(
                "Drag and drop files here",
                type=DataManager.SUPPORTED_FORMATS[upload_type],
                accept_multiple_files=False
            )
            
            if uploaded_file:
                with st.spinner("Processing upload..."):
                    try:
                        data = data_manager.import_data(uploaded_file, upload_type)
                        st.session_state[f"{upload_type.lower()}_data"] = data
                        st.success(f"{upload_type} data loaded successfully!")
                    except Exception as e:
                        st.error(f"Error loading data: {str(e)}")
            
            # Export Options
            st.subheader("Export Results")
            if st.session_state.get('analysis_results'):
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Download Report"):
                        report_path = data_manager.export_results(
                            st.session_state.analysis_results,
                            'report'
                        )
                        with open(report_path, 'rb') as f:
                            st.download_button(
                                "Download Report PDF",
                                f,
                                file_name="analysis_report.pdf",
                                mime="application/pdf"
                            )
                
                with col2:
                    if st.button("Export Raw Data"):
                        data_path = data_manager.export_results(
                            st.session_state.analysis_results,
                            'data'
                        )
                        with open(data_path, 'rb') as f:
                            st.download_button(
                                "Download Data",
                                f,
                                file_name="analysis_data.npz",
                                mime="application/octet-stream"
                            )
    
    with right_col:
        st.markdown("## Analysis Results")
        
        # Mode Correlation Matrix
        st.markdown("Mode Correlation Matrix")
        
        # Create correlation matrix
        modes = ["Chemical-Surface", "Surface-Defect", "Chemical-Defect"]
        matrix = np.array([
            [1.0, 0.7, 0.5],
            [0.7, 1.0, 0.6],
            [0.5, 0.6, 1.0]
        ])
        
        # Plot correlation matrix
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=modes,
            y=modes,
            colorscale='RdBu',
            zmin=-0.5,
            zmax=1
        ))
        
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=50, r=50, t=30, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add tabs for different analysis views
        tab_styles = """
        <style>
            .stTab {
                background-color: transparent;
                color: white;
                padding: 8px 16px;
                border: none;
                border-bottom: 2px solid transparent;
                margin-right: 8px;
            }
            .stTab[data-selected="true"] {
                color: #ff4b4b;
                border-bottom-color: #ff4b4b;
            }
        </style>
        """
        st.markdown(tab_styles, unsafe_allow_html=True)
        
        # Create tabs
        tabs = st.tabs([
            "Chemical-Surface",
            "Surface-Defect", 
            "Chemical-Defect",
            "Photogrammetry"
        ])
        
        # Chemical-Surface tab
        with tabs[0]:
            st.markdown("### Chemical-Surface Analysis")
            x = np.linspace(0, 500, 512)
            y = np.linspace(0, 500, 512)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(X/50) * np.cos(Y/50) + np.random.randn(512, 512) * 0.1
            
            fig_chemical = go.Figure(data=go.Heatmap(
                z=Z,
                colorscale='Viridis',
                colorbar=dict(title="Intensity")
            ))
            
            fig_chemical.update_layout(
                template="plotly_dark",
                height=400,
                margin=dict(l=50, r=50, t=30, b=50)
            )
            
            st.plotly_chart(fig_chemical, use_container_width=True)
        
        # Surface-Defect tab
        with tabs[1]:
            st.markdown("### Surface-Defect Analysis")
            Z_surface = np.exp(-(X**2 + Y**2)/50000) + np.random.randn(512, 512) * 0.05
            
            fig_surface = go.Figure(data=go.Heatmap(
                z=Z_surface,
                colorscale='Viridis',
                colorbar=dict(title="Height")
            ))
            
            fig_surface.update_layout(
                template="plotly_dark",
                height=400,
                margin=dict(l=50, r=50, t=30, b=50)
            )
            
            st.plotly_chart(fig_surface, use_container_width=True)
        
        # Chemical-Defect tab
        with tabs[2]:
            st.markdown("### Chemical-Defect Analysis")
            Z_defect = np.zeros((512, 512))
            # Add some artificial defects
            for _ in range(5):
                x_pos = np.random.randint(0, 512)
                y_pos = np.random.randint(0, 512)
                Z_defect += np.exp(-((X - x_pos*50)**2 + (Y - y_pos*50)**2)/5000)
            
            fig_defect = go.Figure(data=go.Heatmap(
                z=Z_defect,
                colorscale='Viridis',
                colorbar=dict(title="Defect Intensity")
            ))
            
            fig_defect.update_layout(
                template="plotly_dark",
                height=400,
                margin=dict(l=50, r=50, t=30, b=50)
            )
            
            st.plotly_chart(fig_defect, use_container_width=True)
        
        # Photogrammetry tab
        with tabs[3]:
            st.markdown("### Photogrammetry Analysis")
            
            # Image upload section
            uploaded_images = st.file_uploader(
                "Upload images for 3D reconstruction",
                type=['jpg', 'png', 'jpeg'],
                accept_multiple_files=True
            )
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                if uploaded_images:
                    # Process images
                    images = [cv2.imdecode(
                        np.frombuffer(image.read(), np.uint8),
                        cv2.IMREAD_COLOR
                    ) for image in uploaded_images]
                    
                    # Initialize processor
                    processor = PhotogrammetryProcessor()
                    
                    with st.spinner("Processing images..."):
                        results = processor.process_image_sequence(images)
                        
                        if results:
                            # 3D Visualization
                            fig = create_3d_reconstruction_view(
                                results,
                                show_cameras=True,
                                show_features=True
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Animation controls
                            if st.button("Animate View"):
                                animation = create_3d_animation(
                                    results,
                                    speed=1.0
                                )
                                st.plotly_chart(animation, use_container_width=True)
                else:
                    st.info("Upload images to start photogrammetry analysis")
            
            with col2:
                if 'results' in locals() and results:
                    # Analysis metrics
                    st.subheader("Analysis Metrics")
                    
                    st.metric(
                        "Total Features", 
                        len(results['sparse_cloud'])
                    )
                    st.metric(
                        "Dense Points", 
                        len(results['dense_cloud'])
                    )
                    
                    # Quality assessment
                    quality = calculate_reconstruction_quality(results)
                    st.progress(quality)
                    st.caption(f"Reconstruction Quality: {quality:.2%}")
                    
                    # SEHI Integration
                    if 'sehi_data' in st.session_state:
                        st.subheader("SEHI Integration")
                        if st.button("Integrate with SEHI"):
                            integrated_results = processor.integrate_with_sehi(
                                results,
                                st.session_state.sehi_data
                            )
                            if integrated_results:
                                st.success("Integration successful!")
                                st.plotly_chart(
                                    integrated_results['visualization'],
                                    use_container_width=True
                                )

# Add custom CSS
st.markdown("""
<style>
    /* Info icon styling */
    .info-icon {
        margin-left: 5px;
        color: #666;
        cursor: help;
    }
    
    /* Button styling */
    .stButton button {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 4px 8px;
        border-radius: 4px;
        margin: 2px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #1e2530;
        border: none;
    }
    
    /* Slider styling */
    .stSlider {
        padding-top: 0;
    }
    
    /* Header styling */
    h1 {
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def process_multi_modal_analysis(modes, resolution, fusion_method, correlation_threshold, photo_params=None):
    """Process multi-modal analysis based on selected modes."""
    results = {}
    
    # Mode Correlation Matrix
    correlation_matrix = calculate_mode_correlations(modes)
    results['correlation_matrix'] = correlation_matrix
    
    # Process each mode combination
    for mode in modes:
        if mode == "Chemical-Surface":
            results['chemical_surface'] = process_chemical_surface(resolution)
        elif mode == "Chemical-Defect":
            results['chemical_defect'] = process_chemical_defect(resolution)
        elif mode == "Surface-Defect":
            results['surface_defect'] = process_surface_defect(resolution)
        elif mode == "Photogrammetry" and photo_params:
            results['photogrammetry'] = process_photogrammetry(photo_params)
    
    return results

def display_multi_modal_results(results, modes):
    """Display multi-modal analysis results."""
    st.markdown("### Analysis Results")
    
    # Display correlation matrix
    if 'correlation_matrix' in results:
        st.subheader("Mode Correlation Matrix")
        fig_correlation = go.Figure(data=go.Heatmap(
            z=results['correlation_matrix'],
            x=modes,
            y=modes,
            colorscale='RdBu',
            zmid=0
        ))
        fig_correlation.update_layout(
            template="plotly_dark",
            height=400
        )
        st.plotly_chart(fig_correlation, use_container_width=True)
    
    # Display mode-specific results
    for mode in modes:
        st.subheader(f"{mode} Analysis")
        if mode == "Photogrammetry" and 'photogrammetry' in results:
            display_photogrammetry_results(results['photogrammetry'])
        else:
            display_mode_results(results.get(mode.lower().replace("-", "_")))

def display_photogrammetry_results(photo_results):
    """Display photogrammetry analysis results."""
    if not photo_results:
        return
        
    col1, col2 = st.columns(2)
    
    with col1:
        # 3D Visualization
        fig = create_3d_reconstruction_view(photo_results)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Metrics
        st.metric("Total Features", len(photo_results['sparse_cloud']))
        st.metric("Dense Points", len(photo_results['dense_cloud']))
        
        # Quality assessment
        quality = calculate_reconstruction_quality(photo_results)
        st.progress(quality)
        st.caption(f"Reconstruction Quality: {quality:.2%}")

def calculate_mode_correlations(modes):
    """Calculate correlation matrix between different analysis modes."""
    n = len(modes)
    matrix = np.zeros((n, n))
    
    # Example correlation values (replace with actual calculations)
    for i in range(n):
        for j in range(n):
            if i == j:
                matrix[i,j] = 1.0
            else:
                matrix[i,j] = np.random.uniform(0.3, 0.8)
                matrix[j,i] = matrix[i,j]
    
    return matrix

def create_3d_reconstruction_view(results, show_cameras=True, show_features=True):
    """Create interactive 3D visualization of reconstruction."""
    fig = go.Figure()
    
    # Add dense point cloud
    fig.add_trace(go.Scatter3d(
        x=results['dense_cloud'][:, 0],
        y=results['dense_cloud'][:, 1],
        z=results['dense_cloud'][:, 2],
        mode='markers',
        marker=dict(
            size=2,
            color=results['dense_cloud'][:, 3] if results['dense_cloud'].shape[1] > 3 else None,
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Dense Cloud'
    ))
    
    if show_features and 'sparse_cloud' in results:
        # Add feature points
        fig.add_trace(go.Scatter3d(
            x=results['sparse_cloud'][:, 0],
            y=results['sparse_cloud'][:, 1],
            z=results['sparse_cloud'][:, 2],
            mode='markers',
            marker=dict(
                size=3,
                color='red',
                symbol='circle'
            ),
            name='Features'
        ))
    
    # Update layout
    fig.update_layout(
        scene=dict(
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            aspectmode='data'
        ),
        template="plotly_dark",
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    return fig

def calculate_reconstruction_quality(results):
    """Calculate reconstruction quality score."""
    if not results:
        return 0.0
        
    # Feature coverage
    feature_density = len(results['sparse_cloud']) / 1000
    
    # Point cloud density
    cloud_density = len(results['dense_cloud']) / 10000
    
    # Combine metrics
    quality = (min(1.0, feature_density) + min(1.0, cloud_density)) / 2
    
    return quality

def process_chemical_surface(resolution):
    """Process chemical-surface analysis."""
    try:
        # Generate sample data for demonstration
        data = np.zeros((resolution, resolution))
        x, y = np.meshgrid(np.linspace(-5, 5, resolution), np.linspace(-5, 5, resolution))
        data = np.exp(-(x**2 + y**2)/10) + 0.1 * np.random.randn(resolution, resolution)
        return {
            'data': data,
            'x_range': (-5, 5),
            'y_range': (-5, 5),
            'intensity_range': (data.min(), data.max())
        }
    except Exception as e:
        logging.error(f"Error in chemical-surface analysis: {str(e)}")
        return None

def process_chemical_defect(resolution):
    """Process chemical-defect analysis."""
    try:
        # Generate sample defect map
        data = np.zeros((resolution, resolution))
        x, y = np.meshgrid(np.linspace(-5, 5, resolution), np.linspace(-5, 5, resolution))
        # Add some artificial defects
        defects = np.random.rand(3, 2) * 10 - 5
        for dx, dy in defects:
            data += np.exp(-((x - dx)**2 + (y - dy)**2)/0.5)
        return {
            'data': data,
            'defect_locations': defects,
            'intensity_range': (data.min(), data.max())
        }
    except Exception as e:
        logging.error(f"Error in chemical-defect analysis: {str(e)}")
        return None

def process_surface_defect(resolution):
    """Process surface-defect analysis."""
    try:
        # Generate sample surface defect map
        data = np.zeros((resolution, resolution))
        x, y = np.meshgrid(np.linspace(-5, 5, resolution), np.linspace(-5, 5, resolution))
        # Create surface with defects
        surface = np.sin(x) * np.cos(y)
        defects = np.random.rand(5, 2) * 10 - 5
        for dx, dy in defects:
            surface += 0.5 * np.exp(-((x - dx)**2 + (y - dy)**2)/0.2)
        return {
            'data': surface,
            'defect_locations': defects,
            'height_range': (surface.min(), surface.max())
        }
    except Exception as e:
        logging.error(f"Error in surface-defect analysis: {str(e)}")
        return None

def display_mode_results(results):
    """Display results for a specific analysis mode."""
    if not results:
        st.warning("No results available for this mode.")
        return

    # Create visualization based on data type
    if 'data' in results:
        fig = go.Figure(data=go.Heatmap(
            z=results['data'],
            colorscale='Viridis',
            zmin=results['data'].min(),
            zmax=results['data'].max()
        ))
        
        # Add defect markers if available
        if 'defect_locations' in results:
            fig.add_trace(go.Scatter(
                x=results['defect_locations'][:, 0],
                y=results['defect_locations'][:, 1],
                mode='markers',
                marker=dict(
                    size=10,
                    color='red',
                    symbol='x'
                ),
                name='Defects'
            ))
        
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display metrics
        col1, col2 = st.columns(2)
        with col1:
            if 'intensity_range' in results:
                st.metric("Min Intensity", f"{results['intensity_range'][0]:.2f}")
                st.metric("Max Intensity", f"{results['intensity_range'][1]:.2f}")
            if 'height_range' in results:
                st.metric("Min Height", f"{results['height_range'][0]:.2f}")
                st.metric("Max Height", f"{results['height_range'][1]:.2f}")
        
        with col2:
            if 'defect_locations' in results:
                st.metric("Defect Count", len(results['defect_locations']))
                avg_intensity = np.mean(results['data'])
                st.metric("Average Intensity", f"{avg_intensity:.2f}")

def process_photogrammetry(params):
    """Process photogrammetry analysis."""
    try:
        # For demonstration, generate sample point cloud
        n_points = 1000
        points = np.random.randn(n_points, 3)
        intensities = np.random.rand(n_points)
        
        return {
            'sparse_cloud': points[:100],
            'dense_cloud': np.column_stack([points, intensities]),
            'features': {'keypoints': [], 'descriptors': []},
            'camera_poses': np.array([np.eye(4)])
        }
    except Exception as e:
        logging.error(f"Error in photogrammetry processing: {str(e)}")
        return None 