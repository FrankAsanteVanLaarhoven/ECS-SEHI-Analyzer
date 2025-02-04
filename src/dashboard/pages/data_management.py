import streamlit as st
import pandas as pd
import numpy as np
from utils.data_manager import DataManager
from utils.preprocessing import SEHIPreprocessor
from utils.sehi_analysis import SEHIAnalyzer
from utils.photogrammetry import PhotogrammetryProcessor
from utils.visualization_3d import LidarVisualizer
from utils.spectral_visualization import SpectralVisualizer
from utils.surface_analysis import SurfaceAnalyzer
from utils.chemical_analysis import ChemicalAnalyzer
from utils.defect_analysis import DefectAnalyzer
import plotly.graph_objects as go
from pathlib import Path

def create_preview_visualization(data_type, data, file_name):
    """Create preview visualization based on data type."""
    if data_type == "SEHI":
        # SEHI spectrum preview with interactive features
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=data['spectrum'][:100],
            mode='lines',
            name='Spectrum'
        ))
        fig.update_layout(
            title="SEHI Spectrum Preview",
            template="plotly_dark",
            height=300
        )
        return fig
    
    elif data_type == "Photogrammetry":
        # 3D point cloud preview
        if 'sparse_cloud' in data:
            points = data['sparse_cloud']
            fig = go.Figure(data=[
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode='markers',
                    marker=dict(size=2)
                )
            ])
            fig.update_layout(
                title="3D Reconstruction Preview",
                template="plotly_dark",
                height=300
            )
            return fig
    
    elif data_type in ["Surface", "ECS"]:
        # Get view type from session state or initialize it
        if 'view_type' not in st.session_state:
            st.session_state.view_type = "Surface"
            
        # Add view type selection
        view_type = st.radio(
            "View Type",
            ["Surface", "Contour", "Wireframe"],
            key=f"view_type_{file_name}"
        )
        
        # Create interactive 3D surface plot with defect highlighting
        if 'height_map' in data or 'surface_data' in data:
            surface_data = data.get('height_map', data.get('surface_data'))
            x = np.linspace(0, surface_data.shape[0], surface_data.shape[0])
            y = np.linspace(0, surface_data.shape[1], surface_data.shape[1])
            X, Y = np.meshgrid(x, y)

            # Create main surface plot
            fig = go.Figure()
            
            if view_type == "Surface":
                # Add surface plot with colorscale
                fig.add_trace(go.Surface(
                    x=X,
                    y=Y,
                    z=surface_data,
                    colorscale='Viridis',
                    showscale=True,
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
            
            elif view_type == "Contour":
                # Add contour plot
                fig.add_trace(go.Contour(
                    x=x,
                    y=y,
                    z=surface_data,
                    colorscale='Viridis',
                    contours=dict(
                        showlabels=True,
                        labelfont=dict(size=12, color='white')
                    ),
                    line=dict(width=2)
                ))
            
            else:  # Wireframe
                # Add wireframe plot
                fig.add_trace(go.Scatter3d(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=surface_data.flatten(),
                    mode='lines',
                    line=dict(
                        color='cyan',
                        width=1
                    ),
                    name='Wireframe'
                ))

            # Add defect markers if available (for all view types)
            if 'defects' in data:
                defect_positions = data['defects']
                fig.add_trace(go.Scatter3d(
                    x=defect_positions[:, 0],
                    y=defect_positions[:, 1],
                    z=defect_positions[:, 2],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color='red',
                        symbol='sphere',
                        line=dict(color='rgba(255, 0, 0, 0.8)', width=2)
                    ),
                    name='Defects'
                ))

            # Add animation frames for 3D views (Surface and Wireframe)
            if view_type != "Contour":
                frames = []
                for i in range(4):
                    camera = dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(
                            x=1.5 * np.cos(i * np.pi/2),
                            y=1.5 * np.sin(i * np.pi/2),
                            z=1.2
                        )
                    )
                    frames.append(go.Frame(layout=dict(scene_camera=camera)))

                fig.frames = frames

                # Add animation buttons
                fig.update_layout(
                    updatemenus=[{
                        'type': 'buttons',
                        'showactive': False,
                        'buttons': [
                            {
                                'label': 'Play',
                                'method': 'animate',
                                'args': [None, {
                                    'frame': {'duration': 1000, 'redraw': True},
                                    'fromcurrent': True,
                                    'transition': {'duration': 500}
                                }]
                            },
                            {
                                'label': '⟲ Reset',
                                'method': 'animate',
                                'args': [[None], {
                                    'frame': {'duration': 0, 'redraw': True},
                                    'mode': 'immediate',
                                    'transition': {'duration': 0}
                                }]
                            }
                        ]
                    }]
                )

            # Update layout based on view type
            if view_type == "Contour":
                fig.update_layout(
                    title=f"2D Contour Analysis - {data_type}",
                    template="plotly_dark",
                    height=600,
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=True
                )
            else:
                fig.update_layout(
                    title=f"3D {view_type} Analysis - {data_type}",
                    template="plotly_dark",
                    scene=dict(
                        xaxis_title="X (μm)",
                        yaxis_title="Y (μm)",
                        zaxis_title="Height (nm)",
                        camera=dict(
                            up=dict(x=0, y=0, z=1),
                            center=dict(x=0, y=0, z=0),
                            eye=dict(x=1.5, y=1.5, z=1.2)
                        ),
                        aspectmode='data'
                    ),
                    height=600,
                    margin=dict(l=0, r=0, t=30, b=0),
                    showlegend=True
                )

            # Add hover information
            fig.update_traces(
                hoverinfo='x+y+z',
                hovertemplate=(
                    "X: %{x:.1f} μm<br>" +
                    "Y: %{y:.1f} μm<br>" +
                    "Height: %{z:.2f} nm<br>" +
                    "<extra></extra>"
                )
            )

            return fig
    
    elif data_type == "Defect":
        return create_defect_visualization(data, file_name)
    
    return None

def create_defect_visualization(data, file_name):
    """Create enhanced 3D defect visualization with metrics."""
    
    # Create main figure
    fig = go.Figure()
    
    # Add 3D surface plot with defect intensity mapping
    if 'defect_map' in data:
        x = np.linspace(0, data['defect_map'].shape[0], data['defect_map'].shape[0])
        y = np.linspace(0, data['defect_map'].shape[1], data['defect_map'].shape[1])
        X, Y = np.meshgrid(x, y)
        
        # Add main surface with defect intensity coloring
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=data['defect_map'],
            colorscale='RdBu',  # Red-Blue scale for defect intensity
            showscale=True,
            colorbar=dict(
                title="Defect Intensity",
                titleside="right",
                ticks="outside",
                tickfont=dict(size=12),
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
        
        # Add metrics at the top
        metrics = {
            "Total Defects": data.get('total_defects', 1),
            "Average Size": f"{data.get('average_size', 262144.00):.2f} nm",
            "Coverage": f"{data.get('coverage', 100.00):.2f}%",
            "Confidence": f"{data.get('confidence', 100.00):.2f}%"
        }
        
        # Create metric annotations
        metric_text = ""
        for label, value in metrics.items():
            metric_text += f"{label}: {value}     "
        
        fig.add_annotation(
            text=metric_text,
            xref="paper",
            yref="paper",
            x=0,
            y=1.15,
            showarrow=False,
            font=dict(size=14, color="white"),
            bgcolor="rgba(0,0,0,0.5)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            borderpad=10
        )
        
        # Add view control buttons
        buttons = [
            dict(
                args=[{"visible": [True]}],
                label="Both",
                method="restyle"
            ),
            dict(
                args=[{"opacity": 0.5}],
                label="Intensity Only",
                method="restyle"
            ),
            dict(
                args=[{"opacity": 1.0}],
                label="Confidence Only",
                method="restyle"
            )
        ]
        
        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=True,
                    buttons=buttons,
                    x=0.1,
                    y=1.1,
                    xanchor="left",
                    yanchor="top",
                    bgcolor="rgba(0,0,0,0.5)",
                    font=dict(color="white")
                )
            ]
        )
        
        # Add camera controls
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.5, y=1.5, z=1.2)
        )
        
        # Update layout
        fig.update_layout(
            template="plotly_dark",
            scene=dict(
                xaxis_title="X Position (nm)",
                yaxis_title="Y Position (nm)",
                zaxis_title="Height (nm)",
                camera=camera,
                aspectmode='data',
                bgcolor="rgb(17,17,17)"
            ),
            height=800,
            margin=dict(l=0, r=0, t=100, b=0),
            paper_bgcolor="rgb(17,17,17)",
            plot_bgcolor="rgb(17,17,17)",
            showlegend=False
        )
        
        # Add tool buttons (camera, zoom, pan, etc.)
        fig.update_layout(
            modebar=dict(
                bgcolor="rgba(0,0,0,0)",
                color="white",
                activecolor="red"
            ),
            modebar_add=[
                "camera",
                "zoom3d",
                "pan3d",
                "resetcamera3d",
                "toimage",
                "hoverclosest3d"
            ]
        )
        
        # Add hover information
        fig.update_traces(
            hoverinfo="x+y+z+text",
            hovertemplate=(
                "X: %{x:.1f} nm<br>" +
                "Y: %{y:.1f} nm<br>" +
                "Intensity: %{z:.2f}<br>" +
                "<extra></extra>"
            )
        )
        
        return fig
    return None

def render_batch_processing_ui(data_type, files):
    """Render batch processing interface."""
    st.subheader("Batch Processing")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        process_options = {
            "SEHI": ["Baseline Correction", "Peak Detection", "Noise Reduction"],
            "Photogrammetry": ["Feature Detection", "Dense Reconstruction", "Mesh Generation"],
            "Surface": ["Roughness Analysis", "Feature Extraction", "Defect Detection"],
            "Defect": ["Anomaly Detection", "Classification", "Segmentation"]
        }
        
        selected_processes = st.multiselect(
            "Select Processing Steps",
            process_options.get(data_type, [])
        )
    
    with col2:
        if st.button("Process Batch", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(files):
                status_text.text(f"Processing {file.name}...")
                progress_bar.progress((i + 1) / len(files))
                
                try:
                    # Process file with selected options
                    processed_data = process_file_with_options(
                        file, data_type, selected_processes
                    )
                    
                    # Store results
                    if processed_data:
                        st.session_state[f"{data_type.lower()}_batch_{file.name}"] = processed_data
                
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            status_text.text("Batch processing complete!")

def enhance_metadata_display(data):
    """Create enhanced metadata display."""
    if not data.get('metadata'):
        return
    
    metadata = data['metadata']
    
    # Create tabs for different metadata categories
    tabs = st.tabs(["Basic Info", "Parameters", "Statistics", "History"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("File Size", f"{metadata.get('file_size', 0)/1024:.1f} KB")
            st.metric("Dimensions", metadata.get('dimensions', 'N/A'))
        with col2:
            st.metric("Format", metadata.get('format', 'N/A'))
            st.metric("Created", metadata.get('created', 'N/A'))
    
    with tabs[1]:
        st.json(metadata.get('parameters', {}))
    
    with tabs[2]:
        if 'statistics' in metadata:
            stats = metadata['statistics']
            cols = st.columns(len(stats))
            for i, (key, value) in enumerate(stats.items()):
                with cols[i]:
                    st.metric(key, f"{value:.2f}")
    
    with tabs[3]:
        if 'processing_history' in metadata:
            for step in metadata['processing_history']:
                st.markdown(f"- {step}")

def process_file_with_options(file, data_type, selected_processes):
    """Process a file with selected processing options."""
    try:
        if data_type == "SEHI":
            processor = SEHIPreprocessor()
            data = processor.process(file)
            for process in selected_processes:
                if process == "Baseline Correction":
                    data = processor.correct_baseline(data)
                elif process == "Peak Detection":
                    data = processor.detect_peaks(data)
                elif process == "Noise Reduction":
                    data = processor.reduce_noise(data)
            return data
            
        elif data_type == "Photogrammetry":
            processor = PhotogrammetryProcessor()
            data = processor.process_image_sequence([file])
            for process in selected_processes:
                if process == "Feature Detection":
                    data = processor.detect_features(data)
                elif process == "Dense Reconstruction":
                    data = processor.create_dense_cloud(data)
                elif process == "Mesh Generation":
                    data = processor.generate_mesh(data)
            return data
            
        # Add similar processing for other data types
        return None
        
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None

def export_single_file(data, export_format, filename):
    """Export a single file in the specified format."""
    try:
        data_manager = DataManager()
        
        if export_format == "PDF":
            path = data_manager.export_results(data, 'report')
            with open(path, 'rb') as f:
                st.download_button(
                    "Download PDF",
                    f,
                    file_name=f"{filename}.pdf",
                    mime="application/pdf"
                )
        elif export_format == "HTML":
            # Create interactive HTML visualization
            fig = create_preview_visualization(data.get('type', 'unknown'), data, filename)
            if fig:
                html = fig.to_html(include_plotlyjs=True)
                st.download_button(
                    "Download HTML",
                    html,
                    file_name=f"{filename}.html",
                    mime="text/html"
                )
        else:
            path = data_manager.export_results(data, export_format.lower())
            with open(path, 'rb') as f:
                st.download_button(
                    "Download Data",
                    f,
                    file_name=f"{filename}.{export_format.lower()}",
                    mime="application/octet-stream"
                )
                
    except Exception as e:
        st.error(f"Error exporting file: {str(e)}")

def render_data_management():
    """Render the data management page."""
    # Initialize session state if not exists
    if 'current_data_type' not in st.session_state:
        st.session_state.current_data_type = "SEHI"
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'preview_data' not in st.session_state:
        st.session_state.preview_data = {}

    st.markdown("# Data Management")
    
    # Initialize components
    data_manager = DataManager()
    
    # Create two columns for upload and export
    upload_col, export_col = st.columns([2, 1])
    
    with upload_col:
        st.markdown("## Data Upload")
        
        # Data type selection with descriptions
        previous_data_type = st.session_state.current_data_type
        data_type = st.selectbox(
            "Select Data Type",
            ["SEHI", "ECS", "LiDAR", "Photogrammetry", "Chemical", "Surface", "Defect"],
            help="Choose the type of data you want to upload",
            key="data_type_selector"
        )
        
        # Update current data type without triggering rerun
        if data_type != previous_data_type:
            st.session_state.current_data_type = data_type
            
        # Show supported formats and descriptions
        formats = DataManager.SUPPORTED_FORMATS[data_type]
        st.caption(f"Supported formats: {', '.join(formats)}")
        
        # Add format-specific help text
        format_descriptions = {
            "SEHI": "Upload SEHI spectroscopy data for advanced analysis",
            "ECS": "Upload electrochemical scanning data",
            "LiDAR": "Upload point cloud data for 3D analysis",
            "Photogrammetry": "Upload images for 3D reconstruction",
            "Chemical": "Upload chemical mapping data",
            "Surface": "Upload surface topography data",
            "Defect": "Upload defect detection data"
        }
        st.info(format_descriptions[data_type])
        
        # File uploader with drag & drop
        uploaded_files = st.file_uploader(
            "Drag and drop your files here",
            type=formats,
            accept_multiple_files=True,
            key=f"uploader_{data_type}"
        )
        
        # Store uploaded files in session state
        if uploaded_files:
            st.session_state.uploaded_files[data_type] = uploaded_files
        
        # Process and display files from session state
        if data_type in st.session_state.uploaded_files:
            for file in st.session_state.uploaded_files[data_type]:
                file_key = f"{data_type}_{file.name}"
                
                # Only process if not already processed
                if file_key not in st.session_state.preview_data:
                    with st.spinner(f"Processing {file.name}..."):
                        try:
                            # Process data based on type
                            data = process_file(file, data_type, data_manager)
                            if data:
                                st.session_state.preview_data[file_key] = data
                                st.success(f"Successfully loaded: {file.name}")
                        except Exception as e:
                            st.error(f"Error processing {file.name}: {str(e)}")
                
                # Display preview for processed data
                if file_key in st.session_state.preview_data:
                    data = st.session_state.preview_data[file_key]
                    with st.expander(f"Preview: {file.name}"):
                        preview = create_preview_visualization(data_type, data, file.name)
                        if preview is not None:
                            st.plotly_chart(preview, use_container_width=True)
                        
                        # Enhanced metadata display
                        enhance_metadata_display(data)
                        
                        # Export options
                        col1, col2 = st.columns(2)
                        with col1:
                            export_format = st.selectbox(
                                "Export as",
                                ["NPZ", "CSV", "JSON", "PDF", "HTML", "PNG", "XLSX"],
                                key=f"export_format_{file_key}"
                            )
                        with col2:
                            if st.button("Export", key=f"export_{file_key}"):
                                export_single_file(data, export_format, file.name)

def process_file(file, data_type, data_manager):
    """Process a single file based on its type."""
    processors = {
        "SEHI": SEHIPreprocessor(),
        "Photogrammetry": PhotogrammetryProcessor(),
        "LiDAR": LidarVisualizer(),
        "Chemical": ChemicalAnalyzer(),
        "Surface": SurfaceAnalyzer(),
        "Defect": DefectAnalyzer()
    }
    
    if data_type in processors:
        processor = processors[data_type]
        if data_type == "Photogrammetry":
            return processor.process_image_sequence([file])
        else:
            return processor.process_data(file)
    else:
        return data_manager.import_data(file, data_type) 