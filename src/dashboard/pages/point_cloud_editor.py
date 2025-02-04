import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import h5py
import scipy.spatial
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import datetime
import json

# Optional imports with error handling
try:
    import open3d as o3d
except ImportError:
    st.warning("open3d not installed. Some 3D features will be limited.")
    o3d = None

try:
    import trimesh
except ImportError:
    st.warning("trimesh not installed. Some mesh processing features will be limited.")
    trimesh = None

try:
    import laspy
except ImportError:
    st.warning("laspy not installed. LAS/LAZ file support will be limited.")
    laspy = None

# Add new imports for game engine exports
try:
    import unreal  # For Unreal Engine integration
except ImportError:
    unreal = None

try:
    import fbx  # For FBX export
except ImportError:
    fbx = None

# Add new imports for voice capabilities
try:
    import speech_recognition as sr
    import pyttsx3
    from gtts import gTTS
    import sounddevice as sd
    import soundfile as sf
except ImportError:
    st.warning("Voice capabilities limited. Install speech_recognition, pyttsx3, and gtts packages.")

class PointCloudProcessor:
    """Handle point cloud processing with graceful fallbacks."""
    
    def __init__(self):
        self.has_open3d = o3d is not None
        self.has_trimesh = trimesh is not None
        self.has_laspy = laspy is not None
        self.point_cloud = None
        self.processed_cloud = None
        
        # Initialize fallback processor if trimesh is not available
        if not self.has_trimesh:
            st.info("Using fallback processor for mesh operations")
    
    def load_point_cloud(self, file):
        """Load point cloud with format detection and fallbacks."""
        try:
            file_ext = Path(file.name).suffix.lower()
            
            if file_ext == '.pcd' and self.has_open3d:
                self.point_cloud = o3d.io.read_point_cloud(file)
            elif file_ext == '.ply':
                if self.has_trimesh:
                    self.point_cloud = trimesh.load(file)
                else:
                    # Fallback to numpy for PLY files
                    self.point_cloud = self._load_ply_fallback(file)
            elif file_ext == '.xyz':
                # Basic XYZ format using numpy
                data = np.loadtxt(file)
                self.point_cloud = {'points': data[:, :3]}
            else:
                st.error(f"Unsupported file format: {file_ext}")
                return False
            
            return True
            
        except Exception as e:
            st.error(f"Error loading point cloud: {str(e)}")
            return False

    def _load_ply_fallback(self, file):
        """Simple PLY loader using numpy."""
        try:
            # Read PLY header
            header_end = False
            vertex_count = 0
            
            while not header_end:
                line = file.readline().decode().strip()
                if line == "end_header":
                    header_end = True
                elif "element vertex" in line:
                    vertex_count = int(line.split()[-1])
            
            # Read vertex data
            data = np.loadtxt(file, max_rows=vertex_count)
            return {'points': data[:, :3]}
            
        except Exception as e:
            st.error(f"Error in PLY fallback loader: {str(e)}")
            return None

    def process_without_trimesh(self, operation):
        """Handle operations without trimesh."""
        if operation == "mesh":
            st.warning("Mesh processing requires trimesh. Using simplified processing.")
            # Implement basic processing using numpy
            return self._basic_processing()
        
        return None

    def _basic_processing(self):
        """Basic point cloud processing using numpy."""
        if self.point_cloud is None:
            return None
            
        points = self.point_cloud['points']
        
        # Basic cleaning
        # Remove duplicates
        points = np.unique(points, axis=0)
        
        # Remove outliers using statistical method
        distances = np.zeros(len(points))
        mean = np.mean(points, axis=0)
        for i, point in enumerate(points):
            distances[i] = np.linalg.norm(point - mean)
        
        std = np.std(distances)
        mask = distances < (std * 3)  # Keep points within 3 standard deviations
        
        return {'points': points[mask]}

    def export_to_game_engine(self, format_type, export_path):
        """Export point cloud data for game engines."""
        try:
            if format_type == "Unreal":
                return self.export_to_unreal(export_path)
            elif format_type == "Unity":
                return self.export_to_unity(export_path)
            elif format_type == "FBX":
                return self.export_to_fbx(export_path)
            else:
                st.error(f"Unsupported game engine format: {format_type}")
                return False
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
            return False

    def export_to_unreal(self, path):
        """Export to Unreal Engine format."""
        if not self.point_cloud:
            return False
        
        # Convert to Unreal-compatible format
        # Implementation depends on specific Unreal Engine requirements
        pass

    def export_to_unity(self, path):
        """Export to Unity format."""
        if not self.point_cloud:
            return False
        
        # Convert to Unity-compatible format
        # Implementation depends on specific Unity requirements
        pass

    def export_to_fbx(self, path):
        """Export to FBX format."""
        # Implementation for FBX export
        pass

# Add new VR/AR-specific class
class VRProcessor:
    """Handle VR processing with performance optimizations."""
    
    def __init__(self):
        self.chunk_size = 10000  # Points per chunk for performance
        self.max_points_vr = 1000000  # Max points to render in VR
        self.lod_levels = 3  # Levels of detail
        
    def prepare_for_vr(self, points):
        """Prepare point cloud data for VR rendering."""
        try:
            # Downsample if too many points
            if len(points) > self.max_points_vr:
                points = self.downsample_for_vr(points)
            
            # Create LOD levels
            lod_data = self.create_lod_levels(points)
            
            # Chunk data for streaming
            chunks = self.create_chunks(points)
            
            return {
                'points': points,
                'lod_data': lod_data,
                'chunks': chunks
            }
        except Exception as e:
            st.error(f"VR preparation failed: {str(e)}")
            return None
            
    def downsample_for_vr(self, points):
        """Downsample points while preserving features."""
        try:
            # Use voxel grid downsampling
            if o3d is not None:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                downsampled = pcd.voxel_down_sample(voxel_size=0.05)
                return np.asarray(downsampled.points)
            else:
                # Fallback to random sampling
                indices = np.random.choice(len(points), self.max_points_vr, replace=False)
                return points[indices]
        except Exception as e:
            st.warning(f"Using simple downsampling due to: {str(e)}")
            return points[::len(points)//self.max_points_vr]

    def create_lod_levels(self, points):
        """Create levels of detail for progressive loading."""
        lod_data = []
        current_points = points
        
        for level in range(self.lod_levels):
            # Reduce points by half for each level
            n_points = len(current_points) // 2
            if n_points < 1000:  # Minimum point threshold
                break
                
            indices = np.random.choice(len(current_points), n_points, replace=False)
            current_points = current_points[indices]
            lod_data.append(current_points)
            
        return lod_data

    def create_chunks(self, points):
        """Create chunks for streaming data."""
        return [points[i:i + self.chunk_size] for i in range(0, len(points), self.chunk_size)]

class VoiceController:
    """Enhanced voice controller with multi-language support and advanced commands."""
    
    def __init__(self):
        self.recognizer = sr.Recognizer() if 'sr' in globals() else None
        try:
            self.engine = pyttsx3.init()
            # Set properties
            self.engine.setProperty('rate', 150)
            self.engine.setProperty('volume', 0.9)
            
            # Get available voices
            self.voices = self.engine.getProperty('voices')
        except:
            self.engine = None
        
        # Enhanced command set with multiple languages
        self.commands = {
            'en': {
                "zoom in": self.zoom_in,
                "zoom out": self.zoom_out,
                "rotate left": self.rotate_left,
                "rotate right": self.rotate_right,
                "measure distance": self.start_distance_measurement,
                "measure area": self.start_area_measurement,
                "measure volume": self.start_volume_measurement,
                "analyze surface": self.analyze_surface,
                "detect defects": self.detect_defects,
                "start recording": self.start_recording,
                "stop recording": self.stop_recording,
                "save file": self.save_file,
                "export model": self.export_model,
                "switch to vr": self.switch_to_vr,
                "help": self.voice_help
            },
            'es': {
                "acercar": self.zoom_in,
                "alejar": self.zoom_out,
                "girar izquierda": self.rotate_left,
                "girar derecha": self.rotate_right,
                "medir distancia": self.start_distance_measurement,
                # ... (add more Spanish commands)
            },
            'fr': {
                "zoomer": self.zoom_in,
                "dézoomer": self.zoom_out,
                "tourner à gauche": self.rotate_left,
                "tourner à droite": self.rotate_right,
                # ... (add more French commands)
            }
        }
        
        # Current language setting
        self.current_language = 'en'
        
        # Enhanced recording settings
        self.is_recording = False
        self.transcript = []
        self.timestamps = []
        self.speakers = []
        self.confidence_scores = []
        
    def set_language(self, language_code):
        """Set the active language for voice commands."""
        if language_code in self.commands:
            self.current_language = language_code
            # Update TTS voice
            if self.engine:
                for voice in self.voices:
                    if language_code in voice.languages:
                        self.engine.setProperty('voice', voice.id)
                        break
            return True
        return False
    
    def listen_for_command(self):
        """Enhanced command listening with noise reduction and accuracy improvement."""
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                
                st.info(f"Listening for command in {self.current_language}...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                
                # Try multiple recognition services for better accuracy
                try:
                    command = self.recognizer.recognize_google(
                        audio, 
                        language=self.current_language
                    ).lower()
                except:
                    try:
                        command = self.recognizer.recognize_wit(
                            audio, 
                            key=st.secrets["wit_ai_key"]
                        ).lower()
                    except:
                        command = self.recognizer.recognize_sphinx(
                            audio, 
                            language=self.current_language
                        ).lower()
                
                # Check commands in current language
                if command in self.commands[self.current_language]:
                    self.speak(f"Executing: {command}")
                    self.commands[self.current_language][command]()
                else:
                    similar_commands = self.find_similar_commands(command)
                    if similar_commands:
                        self.speak(f"Did you mean: {', '.join(similar_commands)}?")
                    else:
                        self.speak("Command not recognized. Say 'help' for available commands.")
                    
        except Exception as e:
            st.error(f"Voice recognition error: {str(e)}")
    
    def start_recording(self):
        """Enhanced recording with speaker diarization and timestamps."""
        self.is_recording = True
        try:
            with sr.Microphone() as source:
                st.info("Recording... Click 'Stop Recording' when finished.")
                
                while self.is_recording:
                    audio = self.recognizer.listen(source, timeout=None)
                    
                    # Get transcription with metadata
                    result = self.recognizer.recognize_google_cloud(
                        audio,
                        language=self.current_language,
                        show_all=True
                    )
                    
                    if result:
                        # Extract enhanced information
                        text = result['alternatives'][0]['transcript']
                        confidence = result['alternatives'][0]['confidence']
                        timestamp = datetime.datetime.now()
                        
                        # Store all metadata
                        self.transcript.append(text)
                        self.timestamps.append(timestamp)
                        self.confidence_scores.append(confidence)
                        
                        # Real-time display
                        st.text(f"[{timestamp.strftime('%H:%M:%S')}] ({confidence:.2f}): {text}")
                        
        except Exception as e:
            st.error(f"Recording error: {str(e)}")
    
    def stop_recording(self):
        """Enhanced transcript saving with metadata."""
        self.is_recording = False
        
        # Create detailed transcript
        full_transcript = []
        for text, timestamp, confidence in zip(
            self.transcript, 
            self.timestamps, 
            self.confidence_scores
        ):
            full_transcript.append({
                'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'text': text,
                'confidence': confidence
            })
        
        # Display formatted transcript
        st.json(full_transcript)
        
        # Save as JSON
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"transcript_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(full_transcript, f, indent=2)
        
        # Offer download options
        st.download_button(
            label="Download JSON Transcript",
            data=json.dumps(full_transcript, indent=2),
            file_name=filename,
            mime="application/json"
        )
        
        # Also offer plain text version
        plain_text = "\n".join([f"[{t['timestamp']}] {t['text']}" for t in full_transcript])
        st.download_button(
            label="Download Text Transcript",
            data=plain_text,
            file_name=f"transcript_{timestamp}.txt",
            mime="text/plain"
        )
    
    def find_similar_commands(self, command):
        """Find similar commands using fuzzy matching."""
        from difflib import get_close_matches
        available_commands = list(self.commands[self.current_language].keys())
        return get_close_matches(command, available_commands, n=3, cutoff=0.6)

def render_point_cloud_editor():
    """Render advanced point cloud and photogrammetry editor."""
    # Initialize processor
    if 'processor' not in st.session_state:
        st.session_state.processor = PointCloudProcessor()

    st.markdown("""
        <style>
            .point-cloud-container {
                background: #1a2634;
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
            }
            .editor-controls {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                padding: 10px;
                background: #2a3f50;
                border-radius: 8px;
                margin-bottom: 10px;
            }
            .visualization-panel {
                height: 600px;
                background: #000;
                border-radius: 8px;
                position: relative;
            }
            .metrics-panel {
                position: absolute;
                right: 20px;
                top: 20px;
                background: rgba(0,0,0,0.8);
                padding: 15px;
                border-radius: 8px;
                color: #00ff00;
                font-family: monospace;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("Point Cloud & Photogrammetry Editor")

    # Enhanced File Upload Section
    with st.expander("Data Import", expanded=True):
        cols = st.columns([1, 1, 1, 1])
        with cols[0]:
            point_cloud_file = st.file_uploader("Upload Point Cloud", 
                type=['pcd', 'ply', 'xyz', 'las', 'h5', 'pts'])
        with cols[1]:
            image_files = st.file_uploader("Upload Images for Photogrammetry", 
                type=['jpg', 'png', 'tiff'], accept_multiple_files=True)
        with cols[2]:
            reference_file = st.file_uploader("Reference Data", 
                type=['h5', 'csv', 'json'])
        with cols[3]:
            calibration_file = st.file_uploader("Camera Calibration", 
                type=['xml', 'json'])

    # Main Processing Tabs
    tabs = st.tabs([
        "3D Viewer", 
        "Point Cloud Processing", 
        "Photogrammetry",
        "Cloud Compare",
        "Analysis",
        "VR/AR View",
        "Export & Integration"
    ])

    # Enhanced 3D Viewer Tab
    with tabs[0]:
        st.subheader("Interactive 3D Viewer")
        
        # Advanced Viewer Controls
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            view_mode = st.selectbox("View Mode", 
                ["Points", "Surface", "Wireframe", "Textured", "Intensity"])
        with col2:
            point_size = st.slider("Point Size", 1, 20, 3)
        with col3:
            color_by = st.selectbox("Color By", 
                ["Height", "Intensity", "Classification", "RGB", "Normals", "Curvature"])
        with col4:
            visualization_quality = st.select_slider("Quality",
                options=["Fast", "Balanced", "High Quality"])

        # 3D Visualization
        with st.container():
            fig = create_3d_viewer(point_cloud_file, view_mode, point_size, color_by)
            st.plotly_chart(fig, use_container_width=True)

    # Enhanced Point Cloud Processing Tab
    with tabs[1]:
        st.subheader("Point Cloud Processing")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Pre-processing")
            if st.button("Remove Outliers"):
                remove_outliers()
            if st.button("Noise Reduction"):
                reduce_noise()
            if st.button("Downsample"):
                downsample_cloud()
            if st.button("Register Points"):
                register_points()

        with col2:
            st.markdown("### Advanced Processing")
            if st.button("Surface Reconstruction"):
                reconstruct_surface()
            if st.button("Feature Detection"):
                detect_features()
            if st.button("Segmentation"):
                segment_cloud()

        with col3:
            st.markdown("### Analysis")
            if st.button("Calculate Normals"):
                calculate_normals()
            if st.button("Curvature Analysis"):
                analyze_curvature()
            if st.button("Roughness Analysis"):
                analyze_roughness()

    # Enhanced Cloud Compare Tab
    with tabs[3]:
        st.subheader("Cloud Compare Tools")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Comparison Methods")
            compare_method = st.selectbox("Compare Method", [
                "C2C Distance",
                "M3C2 Distance",
                "C2M Distance",
                "Hausdorff Distance",
                "Normal-based Comparison",
                "Curvature Comparison",
                "Density Analysis"
            ])
            advanced_options = st.expander("Advanced Options")
            with advanced_options:
                max_distance = st.slider("Max Distance", 0.01, 1.0, 0.1)
                num_threads = st.slider("Threads", 1, 8, 4)
                
            if st.button("Compare Clouds"):
                compare_point_clouds(compare_method, max_distance, num_threads)
            
        with col2:
            st.markdown("### Registration Methods")
            registration_method = st.selectbox("Registration Method", [
                "Global ICP",
                "Point-to-Plane ICP",
                "Colored ICP",
                "Fast Global Registration",
                "Feature-based Registration",
                "Multi-scale ICP",
                "Super4PCS"
            ])
            reg_options = st.expander("Registration Options")
            with reg_options:
                max_iterations = st.slider("Max Iterations", 10, 200, 50)
                ransac_n = st.slider("RANSAC N", 3, 20, 4)
                
            if st.button("Register Clouds"):
                register_clouds(registration_method, max_iterations, ransac_n)

        with col3:
            st.markdown("### Analysis Tools")
            analysis_method = st.selectbox("Analysis Type", [
                "Surface Change Detection",
                "Deformation Analysis",
                "Volume Calculation",
                "Cross Sections",
                "Contour Generation"
            ])
            if st.button("Run Analysis"):
                analyze_clouds(analysis_method)

    # Photogrammetry Tab
    with tabs[2]:
        st.subheader("Photogrammetry Processing")
        
        if image_files:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Image Processing")
                quality = st.select_slider("Quality", 
                    options=["Draft", "Low", "Medium", "High", "Ultra"])
                if st.button("Generate Point Cloud"):
                    generate_point_cloud_from_images(image_files, quality)

            with col2:
                st.markdown("### Export Options")
                export_format = st.selectbox("Export Format", 
                    ["PLY", "PCD", "LAS", "XYZ"])
                if st.button("Export"):
                    export_processed_data(export_format)

    # Analysis Tab
    with tabs[4]:
        st.subheader("Analysis & Measurements")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Measurements")
            measure_type = st.selectbox("Measure Type", 
                ["Distance", "Area", "Volume", "Angle"])
            if st.button("Start Measurement"):
                start_measurement(measure_type)

        with col2:
            st.markdown("### Statistics")
            if st.button("Calculate Statistics"):
                calculate_statistics()

    # VR/AR Tab
    with tabs[5]:
        st.subheader("VR/AR Visualization")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### VR Controls")
            if st.button("Launch VR Mode"):
                launch_vr_mode()
            vr_quality = st.select_slider("VR Quality", 
                options=["Performance", "Balanced", "Quality"])

        with col2:
            st.markdown("### AR Controls")
            if st.button("Launch AR Mode"):
                launch_ar_mode()
            ar_tracking = st.checkbox("Enable Surface Tracking")

    # Export & Integration Tab
    with tabs[6]:
        st.subheader("Export & Game Engine Integration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Standard Exports")
            export_format = st.selectbox("Export Format", [
                "PLY (Binary)",
                "PLY (ASCII)",
                "PCD",
                "XYZ",
                "LAS/LAZ",
                "E57",
                "OBJ + MTL",
                "FBX",
                "GLTF/GLB",
                "Alembic"
            ])
            
            export_options = st.expander("Export Options")
            with export_options:
                include_normals = st.checkbox("Include Normals", value=True)
                include_colors = st.checkbox("Include Colors", value=True)
                include_texture = st.checkbox("Include Texture Maps", value=True)
                compression = st.slider("Compression Level", 0, 9, 6)
                
            if st.button("Export Data"):
                export_data(export_format)

        with col2:
            st.markdown("### Game Engine Integration")
            engine_format = st.selectbox("Target Engine", [
                "Unreal Engine",
                "Unity",
                "Godot",
                "Custom Engine"
            ])
            
            engine_options = st.expander("Engine Options")
            with engine_options:
                scale_factor = st.number_input("Scale Factor", 0.001, 1000.0, 1.0)
                coordinate_system = st.selectbox("Coordinate System", [
                    "Right-Handed (Y-up)",
                    "Right-Handed (Z-up)",
                    "Left-Handed (Y-up)",
                    "Left-Handed (Z-up)"
                ])
                level_of_detail = st.slider("LOD Levels", 1, 5, 3)
                
            if st.button("Prepare for Game Engine"):
                prepare_for_game_engine(engine_format)

    # Add Voice Control Section
    with st.expander("Voice Controls & Accessibility"):
        if 'voice_controller' not in st.session_state:
            st.session_state.voice_controller = VoiceController()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Voice Commands")
            if st.button("🎤 Start Voice Command"):
                st.session_state.voice_controller.listen_for_command()
            if st.button("❓ Voice Help"):
                st.session_state.voice_controller.voice_help()
        
        with col2:
            st.markdown("### Voice Recording")
            if st.button("⏺️ Start Recording"):
                st.session_state.voice_controller.start_recording()
            if st.button("⏹️ Stop Recording"):
                st.session_state.voice_controller.stop_recording()
        
        with col3:
            st.markdown("### Text-to-Speech")
            text_to_read = st.text_area("Enter text to read")
            if st.button("🔊 Read Text"):
                st.session_state.voice_controller.read_documentation(text_to_read)

def create_3d_viewer(point_cloud_file, view_mode, point_size, color_by):
    """Create interactive 3D viewer with error handling."""
    try:
        if point_cloud_file is None:
            return create_empty_plot()
                
        if not st.session_state.processor.load_point_cloud(point_cloud_file):
            return create_empty_plot()
        
        points = st.session_state.processor.point_cloud
        
        # Create Plotly figure
        fig = go.Figure(data=[
            go.Scatter3d(
                x=points['points'][:, 0],
                y=points['points'][:, 1],
                z=points['points'][:, 2],
                mode='markers',
                marker=dict(
                    size=point_size,
                    color=get_color_values(points, color_by),
                    colorscale='Viridis',
                )
            )
        ])
        
        # Update layout
        fig.update_layout(
            scene=dict(
                aspectmode='data',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating 3D view: {str(e)}")
        return create_empty_plot()

def create_empty_plot():
    """Create empty plot with instructions."""
    fig = go.Figure()
    fig.update_layout(
        annotations=[dict(
            text="Upload a point cloud file to view",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False
        )],
        margin=dict(l=0, r=0, t=0, b=0)
    )
    return fig

def get_color_values(points, color_by):
    """Get color values based on selected attribute."""
    try:
        if color_by == "Height":
            return points['points'][:, 2]
        elif color_by == "Intensity" and 'intensity' in points:
            return points['intensity']
        elif color_by == "Classification" and 'classification' in points:
            return points['classification']
        else:
            return points['points'][:, 2]  # Default to height
    except Exception as e:
        st.warning(f"Error getting colors: {str(e)}. Defaulting to height.")
        return points['points'][:, 2]

def process_point_cloud(file):
    """Process uploaded point cloud data."""
    # Implementation for point cloud processing
    pass

def generate_point_cloud_from_images(images, quality):
    """Generate point cloud from photogrammetry images."""
    # Implementation for photogrammetry
    pass

def launch_vr_mode():
    """Launch enhanced VR visualization mode."""
    try:
        # Initialize VR processor if not exists
        if 'vr_processor' not in st.session_state:
            st.session_state.vr_processor = VRProcessor()
        
        # Prepare point cloud data for VR
        if st.session_state.processor.point_cloud is not None:
            vr_data = st.session_state.vr_processor.prepare_for_vr(
                st.session_state.processor.point_cloud['points']
            )
            
            if vr_data is None:
                return
            
            # Update session state
            st.session_state.vr_points = vr_data['points'].tolist()
            st.session_state.vr_lod = vr_data['lod_data']
            st.session_state.vr_chunks = vr_data['chunks']
        
        components.html("""
            <script src="https://aframe.io/releases/1.2.0/aframe.min.js"></script>
            <script src="https://unpkg.com/aframe-point-cloud-component/dist/aframe-point-cloud-component.min.js"></script>
            <script src="https://cdn.jsdelivr.net/gh/donmccurdy/aframe-extras@v6.1.1/dist/aframe-extras.min.js"></script>
            
            <div class="vr-container">
                <a-scene embedded physics="debug: true" renderer="antialias: true; logarithmicDepthBuffer: true">
                    <!-- Environment -->
                    <a-sky color="#ECECEC"></a-sky>
                    <a-grid ground></a-grid>
                    
                    <!-- Point Cloud Container with LOD -->
                    <a-entity id="point-cloud-container" position="0 1.6 -1">
                        <a-entity id="point-cloud-lod"></a-entity>
                    </a-entity>
                    
                    <!-- Enhanced VR Controls -->
                    <a-entity id="leftHand" hand-controls="hand: left" laser-controls>
                        <a-entity id="left-menu" position="0.1 0 0">
                            <a-plane class="menu-item clickable" color="#444" height="0.1" width="0.2" 
                                    position="0 0.15 0" text="value: Measure; align: center"></a-plane>
                            <a-plane class="menu-item clickable" color="#444" height="0.1" width="0.2" 
                                    position="0 0 0" text="value: Select; align: center"></a-plane>
                            <a-plane class="menu-item clickable" color="#444" height="0.1" width="0.2" 
                                    position="0 -0.15 0" text="value: Reset; align: center"></a-plane>
                        </a-entity>
                    </a-entity>
                    
                    <!-- Camera Rig with Teleport -->
                    <a-entity id="rig" movement-controls="fly: true">
                        <a-camera id="camera" position="0 1.6 0">
                            <a-cursor></a-cursor>
                            <a-entity id="hud" position="0 0 -1">
                                <a-text id="measurement-text" value="" align="center" 
                                       position="0 -0.5 0" scale="0.5 0.5 0.5"></a-text>
                            </a-entity>
                        </a-camera>
                    </a-entity>
                </a-scene>
            </div>
            
            <script>
                // Enhanced VR interaction handling
                AFRAME.registerComponent('vr-manager', {
                    init: function() {
                        this.setupLOD();
                        this.setupInteractions();
                        this.setupPerformance();
                    },
                    
                    setupLOD: function() {
                        const lodLevels = ${st.session_state.vr_lod};
                        let currentLOD = 0;
                        
                        // Progressive loading
                        const loadNextLOD = () => {
                            if (currentLOD < lodLevels.length) {
                                this.updatePointCloud(lodLevels[currentLOD]);
                                currentLOD++;
                                setTimeout(loadNextLOD, 1000);
                            }
                        };
                        
                        loadNextLOD();
                    },
                    
                    setupPerformance: function() {
                        // Frustum culling
                        this.el.sceneEl.renderer.setAnimationLoop(() => {
                            // Implement view frustum culling
                        });
                    }
                });
            </script>
            
            <style>
                .vr-container {
                    width: 100%;
                    height: 600px;
                    border-radius: 8px;
                    overflow: hidden;
                }
            </style>
        """, height=650)
        
    except Exception as e:
        st.error(f"Error launching VR mode: {str(e)}")

def launch_ar_mode():
    """Launch AR visualization mode."""
    try:
        components.html("""
            <script src="https://raw.githack.com/AR-js-org/AR.js/master/aframe/build/aframe-ar.js"></script>
            <div class="ar-container">
                <a-scene embedded arjs="sourceType: webcam; debugUIEnabled: false;">
                    <a-marker preset="hiro">
                        <a-entity id="ar-point-cloud"></a-entity>
                    </a-marker>
                    <a-entity camera></a-entity>
                </a-scene>
            </div>
            <style>
                .ar-container {
                    width: 100%;
                    height: 600px;
                    border-radius: 8px;
                    overflow: hidden;
                }
            </style>
        """, height=650)
        
    except Exception as e:
        st.error(f"Error launching AR mode: {str(e)}")

def compare_point_clouds(method, max_distance, num_threads):
    """Advanced point cloud comparison."""
    try:
        # Implementation based on selected method
        if method == "C2C Distance":
            return compute_c2c_distance()
        elif method == "M3C2 Distance":
            return compute_m3c2_distance()
        # ... implement other methods
    except Exception as e:
        st.error(f"Comparison failed: {str(e)}")

def register_clouds(method, max_iterations, ransac_n):
    """Advanced point cloud registration."""
    try:
        if method == "Global ICP":
            return global_icp_registration()
        elif method == "Point-to-Plane ICP":
            return point_to_plane_registration()
        # ... implement other methods
    except Exception as e:
        st.error(f"Registration failed: {str(e)}")

def reconstruct_surface(method, depth, scale):
    """Advanced surface reconstruction."""
    try:
        if method == "Poisson":
            return poisson_reconstruction(depth)
        elif method == "Ball Pivoting":
            return ball_pivoting_reconstruction(scale)
        # ... implement other methods
    except Exception as e:
        st.error(f"Reconstruction failed: {str(e)}")

def analyze_clouds(method):
    """Advanced cloud analysis."""
    try:
        if method == "Surface Change Detection":
            return detect_surface_changes()
        elif method == "Deformation Analysis":
            return analyze_deformation()
        # ... implement other methods
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")

# Implement specific algorithms
def compute_c2c_distance():
    """Compute cloud-to-cloud distance."""
    pass

def compute_m3c2_distance():
    """Compute M3C2 distance."""
    pass

def global_icp_registration():
    """Perform global ICP registration."""
    pass

def point_to_plane_registration():
    """Perform point-to-plane ICP registration."""
    pass

def poisson_reconstruction(depth):
    """Perform Poisson surface reconstruction."""
    pass

def ball_pivoting_reconstruction(scale):
    """Perform ball pivoting surface reconstruction."""
    pass

def detect_surface_changes():
    """Detect changes between surfaces."""
    pass

def analyze_deformation():
    """Analyze surface deformation."""
    pass

def analyze_curvature():
    """Implement curvature analysis."""
    pass

def analyze_roughness():
    """Implement roughness analysis."""
    pass

def export_data(format_type):
    """Handle data export with progress tracking."""
    try:
        with st.spinner(f"Exporting to {format_type}..."):
            progress_bar = st.progress(0)
            
            # Create export path
            export_path = Path(st.session_state.get('export_path', '.'))
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"point_cloud_export_{timestamp}"
            
            # Handle different export formats
            if format_type.startswith("PLY"):
                export_ply(export_path / f"{filename}.ply", 
                          binary='Binary' in format_type)
            elif format_type == "FBX":
                export_fbx(export_path / f"{filename}.fbx")
            # ... handle other formats
            
            progress_bar.progress(100)
            
            # Create download button
            with open(export_path / filename, 'rb') as f:
                st.download_button(
                    label="Download Exported File",
                    data=f,
                    file_name=filename,
                    mime="application/octet-stream"
                )
            
            st.success(f"Export completed: {filename}")
            
    except Exception as e:
        st.error(f"Export failed: {str(e)}")

def prepare_for_game_engine(engine_type):
    """Prepare point cloud data for game engine integration."""
    try:
        with st.spinner(f"Preparing for {engine_type}..."):
            if engine_type == "Unreal Engine":
                # Prepare for Unreal Engine
                prepare_for_unreal()
            elif engine_type == "Unity":
                # Prepare for Unity
                prepare_for_unity()
            # ... handle other engines
            
            st.success(f"Data prepared for {engine_type}")
            
    except Exception as e:
        st.error(f"Game engine preparation failed: {str(e)}")

def prepare_for_unreal():
    """Prepare data specifically for Unreal Engine."""
    # Implementation for Unreal Engine preparation
    pass

def prepare_for_unity():
    """Prepare data specifically for Unity."""
    # Implementation for Unity preparation
    pass

def export_ply(path, binary=True):
    """Export to PLY format."""
    # Implementation for PLY export
    pass

def export_fbx(path):
    """Export to FBX format."""
    # Implementation for FBX export
    pass

def start_measurement(measure_type):
    """Start measurement based on selected type."""
    try:
        if not st.session_state.processor.point_cloud:
            st.error("Please load a point cloud first")
            return

        st.info(f"Click points on the point cloud to measure {measure_type}")
        
        if measure_type == "Distance":
            measure_distance()
        elif measure_type == "Area":
            measure_area()
        elif measure_type == "Volume":
            measure_volume()
        elif measure_type == "Angle":
            measure_angle()
            
    except Exception as e:
        st.error(f"Measurement failed: {str(e)}")

def measure_distance():
    """Measure distance between two points."""
    if 'measurement_points' not in st.session_state:
        st.session_state.measurement_points = []
    
    point = st.session_state.processor.point_cloud
    if len(st.session_state.measurement_points) < 2:
        st.session_state.measurement_points.append(point)
    
    if len(st.session_state.measurement_points) == 2:
        p1, p2 = st.session_state.measurement_points
        distance = np.linalg.norm(p1 - p2)
        st.success(f"Distance: {distance:.3f} units")
        st.session_state.measurement_points = []

def measure_area():
    """Measure area of selected polygon."""
    if 'area_points' not in st.session_state:
        st.session_state.area_points = []
    
    point = st.session_state.processor.point_cloud
    st.session_state.area_points.append(point)
    
    if len(st.session_state.area_points) >= 3:
        points = np.array(st.session_state.area_points)
        area = calculate_polygon_area(points)
        st.success(f"Area: {area:.3f} square units")
        
        if st.button("Finish Area Measurement"):
            st.session_state.area_points = []

def measure_volume():
    """Measure volume of selected region."""
    if 'volume_points' not in st.session_state:
        st.session_state.volume_points = []
    
    point = st.session_state.processor.point_cloud
    st.session_state.volume_points.append(point)
    
    if len(st.session_state.volume_points) >= 4:
        points = np.array(st.session_state.volume_points)
        volume = calculate_volume(points)
        st.success(f"Volume: {volume:.3f} cubic units")
        
        if st.button("Finish Volume Measurement"):
            st.session_state.volume_points = []

def measure_angle():
    """Measure angle between three points."""
    if 'angle_points' not in st.session_state:
        st.session_state.angle_points = []
    
    point = st.session_state.processor.point_cloud
    st.session_state.angle_points.append(point)
    
    if len(st.session_state.angle_points) == 3:
        p1, p2, p3 = st.session_state.angle_points
        angle = calculate_angle(p1, p2, p3)
        st.success(f"Angle: {angle:.1f} degrees")
        st.session_state.angle_points = []

def calculate_polygon_area(points):
    """Calculate area of a polygon using shoelace formula."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def calculate_volume(points):
    """Calculate volume of a 3D region."""
    hull = scipy.spatial.ConvexHull(points)
    return hull.volume

def calculate_angle(p1, p2, p3):
    """Calculate angle between three points."""
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(cos_angle) * 180 / np.pi

def calculate_statistics():
    """Calculate statistical measurements of the point cloud."""
    try:
        if not st.session_state.processor.point_cloud:
            st.error("Please load a point cloud first")
            return

        points = st.session_state.processor.point_cloud['points']
        
        stats = {
            "Number of Points": len(points),
            "Bounding Box": {
                "Min": points.min(axis=0),
                "Max": points.max(axis=0)
            },
            "Mean": points.mean(axis=0),
            "Std Dev": points.std(axis=0)
        }
        
        st.write("### Point Cloud Statistics")
        st.json(stats)
        
    except Exception as e:
        st.error(f"Statistics calculation failed: {str(e)}") 