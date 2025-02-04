import streamlit as st
import plotly.graph_objects as go
import numpy as np
import cv2
from utils.photogrammetry import PhotogrammetryProcessor, PhotogrammetryParameters
from utils.visualization_3d import create_3d_animation

def render_photogrammetry_view():
    """Render the photogrammetry analysis page with interactive 3D visualizations."""
    st.markdown('<h1 class="main-header">Photogrammetry Analysis</h1>', unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Analysis Parameters")
        params = PhotogrammetryParameters(
            min_features=st.slider("Minimum Features", 500, 2000, 1000),
            feature_quality=st.slider("Feature Quality", 0.01, 0.1, 0.01),
            matching_distance=st.slider("Matching Distance", 0.5, 0.9, 0.7),
            ransac_threshold=st.slider("RANSAC Threshold", 1.0, 10.0, 4.0)
        )
        
        st.subheader("Visualization Options")
        show_cameras = st.checkbox("Show Camera Positions", True)
        show_features = st.checkbox("Show Feature Points", True)
        animation_speed = st.slider("Animation Speed", 0.1, 2.0, 1.0)

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Image upload section
        st.subheader("Upload Images")
        uploaded_images = st.file_uploader(
            "Upload multiple images for 3D reconstruction",
            type=['jpg', 'png', 'jpeg'],
            accept_multiple_files=True,
            help="Upload a sequence of images taken from different angles"
        )
        
        if uploaded_images:
            # Process images
            images = [cv2.imdecode(
                np.frombuffer(image.read(), np.uint8),
                cv2.IMREAD_COLOR
            ) for image in uploaded_images]
            
            # Initialize processor with parameters
            processor = PhotogrammetryProcessor(params)
            
            with st.spinner("Processing images..."):
                # Process image sequence
                results = processor.process_image_sequence(images)
                
                if results:
                    # Create interactive 3D visualization
                    fig = create_3d_reconstruction_view(
                        results,
                        show_cameras=show_cameras,
                        show_features=show_features
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add animation controls
                    if st.button("Play Animation"):
                        animation = create_3d_animation(
                            results,
                            speed=animation_speed
                        )
                        st.plotly_chart(animation, use_container_width=True)
    
    with col2:
        if uploaded_images:
            # Analysis results and metrics
            st.subheader("Analysis Results")
            
            # Feature matching statistics
            st.metric("Total Features", len(results['sparse_cloud']))
            st.metric("Dense Points", len(results['dense_cloud']))
            
            # Quality metrics
            reconstruction_quality = calculate_reconstruction_quality(results)
            st.progress(reconstruction_quality)
            st.caption("Reconstruction Quality")
            
            # Display individual images with detected features
            st.subheader("Feature Detection")
            selected_image = st.selectbox(
                "Select Image",
                range(len(images)),
                format_func=lambda x: f"Image {x+1}"
            )
            
            if selected_image is not None:
                display_feature_detection(
                    images[selected_image],
                    results['features'][selected_image]
                )

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
            color=results['dense_cloud'][:, 3],
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Dense Cloud'
    ))
    
    if show_features:
        # Add sparse feature points
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
            name='Feature Points'
        ))
    
    if show_cameras:
        # Add camera positions
        cameras = results['camera_poses']
        fig.add_trace(go.Scatter3d(
            x=cameras[:, 0, 3],
            y=cameras[:, 1, 3],
            z=cameras[:, 2, 3],
            mode='markers+text',
            marker=dict(
                size=8,
                color='blue',
                symbol='square'
            ),
            text=[f"Camera {i+1}" for i in range(len(cameras))],
            name='Cameras'
        ))
    
    # Update layout for better visualization
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
    """Calculate overall reconstruction quality score."""
    # Implement quality metrics
    feature_coverage = min(1.0, len(results['sparse_cloud']) / 1000)
    density_score = min(1.0, len(results['dense_cloud']) / 10000)
    
    return (feature_coverage + density_score) / 2

def display_feature_detection(image, features):
    """Display image with detected features."""
    # Draw features on image
    vis_image = image.copy()
    for kp in features['keypoints']:
        pt = tuple(map(int, kp.pt))
        cv2.circle(vis_image, pt, 3, (0, 255, 0), -1)
    
    st.image(vis_image, channels="BGR", use_column_width=True) 