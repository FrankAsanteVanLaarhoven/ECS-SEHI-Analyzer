import os
import json
import pickle
import datetime
from pathlib import Path
from typing import Dict, List, Union, Optional, Any
import numpy as np
import pandas as pd
import h5py
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
import plotly.express as px
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go

class ModelManager:
    """Manages model training, data, and results storage."""
    
    def __init__(self):
        self.base_path = Path("data")
        self.models_path = self.base_path / "models"
        self.samples_path = self.base_path / "samples"
        self.results_path = self.base_path / "results"
        self._initialize_directories()
        self.sample_datasets = self._load_sample_datasets()
        self.models = {}
        self.model_configs = {}
        self.training_history = {}

    def _initialize_directories(self):
        """Create necessary directories."""
        for path in [self.models_path, self.samples_path, self.results_path]:
            path.mkdir(parents=True, exist_ok=True)

    def render_workspace(self):
        """Render the model management workspace."""
        st.title("🔬 Model Workspace")

        # Sidebar for workspace navigation
        with st.sidebar:
            workspace_mode = st.radio(
                "Workspace Mode",
                ["Sample Data", "Upload Data", "Train Model", "Saved Models", "Results"]
            )

        if workspace_mode == "Sample Data":
            self._render_sample_data_section()
        elif workspace_mode == "Upload Data":
            self._render_upload_section()
        elif workspace_mode == "Train Model":
            self._render_training_section()
        elif workspace_mode == "Saved Models":
            self._render_saved_models()
        else:
            self._render_results_section()

    def _load_sample_datasets(self) -> Dict:
        """Load sample datasets for different analysis types."""
        return {
            'particle': self._generate_particle_sample(),
            'surface': self._generate_surface_sample(),
            'composition': self._generate_composition_sample(),
            'spectral': self._generate_spectral_sample()
        }

    def _generate_particle_sample(self) -> Dict:
        """Generate realistic particle analysis sample data."""
        n_particles = 1000
        
        # Generate particle sizes with log-normal distribution
        sizes = np.random.lognormal(mean=3.0, sigma=0.4, size=n_particles)
        
        # Generate particle positions
        x = np.random.uniform(0, 100, n_particles)
        y = np.random.uniform(0, 100, n_particles)
        
        # Generate composition data (multiple elements)
        compositions = {
            'Pt': np.random.normal(0.7, 0.1, n_particles).clip(0, 1),
            'Pd': np.random.normal(0.2, 0.05, n_particles).clip(0, 1),
            'Au': np.random.normal(0.1, 0.02, n_particles).clip(0, 1)
        }
        
        # Normalize compositions to sum to 1
        total = sum(compositions.values())
        compositions = {k: v/total for k, v in compositions.items()}
        
        # Generate morphology data
        morphology = {
            'sphericity': np.random.normal(0.9, 0.05, n_particles).clip(0, 1),
            'aspect_ratio': np.random.normal(1.1, 0.1, n_particles).clip(0.5, 2),
            'surface_roughness': np.random.normal(0.2, 0.05, n_particles).clip(0, 1)
        }
        
        # Generate spectral signatures
        wavelengths = np.linspace(300, 800, 50)
        spectral_data = np.zeros((n_particles, len(wavelengths)))
        
        # Define characteristic peaks for each element
        peaks = {
            'Pt': [(400, 100), (600, 80)],  # (wavelength, intensity)
            'Pd': [(450, 90), (650, 70)],
            'Au': [(500, 85), (700, 75)]
        }
        
        for i in range(n_particles):
            spectrum = np.zeros(len(wavelengths))
            # Add peaks for each element
            for element, element_peaks in peaks.items():
                concentration = compositions[element][i]
                for peak_wavelength, peak_intensity in element_peaks:
                    spectrum += concentration * peak_intensity * np.exp(
                        -(wavelengths - peak_wavelength)**2 / 1000
                    )
            # Add noise
            spectrum += np.random.normal(0, 2, len(wavelengths))
            spectral_data[i] = spectrum
        
        return {
            'metadata': {
                'sample_type': 'Catalyst Particles',
                'acquisition_date': datetime.datetime.now().strftime('%Y-%m-%d'),
                'instrument': 'TEM-SEHI',
                'resolution': '0.5nm',
                'total_particles': n_particles
            },
            'particle_data': {
                'sizes': sizes.tolist(),  # Convert to list for JSON serialization
                'positions': np.column_stack((x, y)).tolist(),
                'compositions': {
                    k: v.tolist() for k, v in compositions.items()
                },
                'morphology': {
                    k: v.tolist() for k, v in morphology.items()
                }
            },
            'spectral_data': {
                'wavelengths': wavelengths.tolist(),
                'intensities': spectral_data.tolist()
            },
            'statistics': {
                'mean_size': float(np.mean(sizes)),
                'size_std': float(np.std(sizes)),
                'total_particles': int(n_particles),
                'density': float(n_particles / (100 * 100)),  # particles per µm²
                'composition_means': {
                    k: float(np.mean(v)) for k, v in compositions.items()
                },
                'morphology_means': {
                    k: float(np.mean(v)) for k, v in morphology.items()
                }
            }
        }

    def _generate_surface_sample(self) -> Dict:
        """Generate sample surface data."""
        width, height = 100, 100
        
        # Create base surface
        x = np.linspace(0, 10, width)
        y = np.linspace(0, 10, height)
        X, Y = np.meshgrid(x, y)
        
        # Generate realistic surface features
        surface = (
            2.0 * np.sin(X) * np.cos(Y) +                    # Base pattern
            0.5 * np.sin(4*X) * np.cos(4*Y) +                # Fine features
            1.0 * np.exp(-((X-5)**2 + (Y-5)**2)/10) +       # Central peak
            0.2 * np.random.normal(0, 1, (height, width))    # Random roughness
        )
        
        # Normalize surface
        surface = (surface - surface.min()) / (surface.max() - surface.min())
        
        return {
            'type': 'surface_analysis',
            'data': {
                'surface_map': surface,
                'visualization': surface,  # Same as surface_map for 2D data
                'dimensions': (width, height),
                'x_coords': x,
                'y_coords': y
            },
            'metadata': {
                'sample_type': 'Surface Topography',
                'units': 'nm',
                'resolution': f"{width}x{height}",
                'features': [
                    'Base periodic pattern',
                    'Fine surface features',
                    'Central peak feature',
                    'Random surface roughness'
                ]
            },
            'statistics': {
                'mean_height': float(np.mean(surface)),
                'max_height': float(np.max(surface)),
                'min_height': float(np.min(surface)),
                'roughness': float(np.std(surface)),
                'peak_to_valley': float(np.max(surface) - np.min(surface))
            }
        }

    def _generate_composition_sample(self) -> Dict:
        """Generate sample composition mapping data."""
        width, height = 100, 100
        n_elements = 3
        
        # Generate composition maps
        composition_maps = np.zeros((height, width, n_elements))
        element_names = ['Pt', 'Pd', 'Au']
        
        # Create spatially correlated patterns
        for i in range(n_elements):
            base = np.random.normal(0.5, 0.1, (height, width))
            composition_maps[:,:,i] = gaussian_filter(base, sigma=5)
        
        # Normalize to ensure compositions sum to 1
        composition_sum = np.sum(composition_maps, axis=2, keepdims=True)
        composition_maps = composition_maps / composition_sum
        
        # Generate RGB visualization
        rgb_map = np.zeros((height, width, 3))
        colors = [(1,0,0), (0,1,0), (0,0,1)]  # RGB colors for each element
        
        for i in range(n_elements):
            for c in range(3):
                rgb_map[:,:,c] += composition_maps[:,:,i] * colors[i][c]
        
        # Normalize RGB map
        rgb_map = rgb_map / rgb_map.max()
        
        return {
            'composition_maps': composition_maps,
            'elements': element_names,
            'visualization': rgb_map,
            'metadata': {
                'dimensions': (height, width),
                'n_elements': n_elements
            },
            'statistics': {
                'mean_compositions': {
                    elem: float(np.mean(composition_maps[:,:,i]))
                    for i, elem in enumerate(element_names)
                },
                'std_compositions': {
                    elem: float(np.std(composition_maps[:,:,i]))
                    for i, elem in enumerate(element_names)
                }
            }
        }

    def _generate_spectral_sample(self) -> Dict:
        """Generate sample spectral mapping data."""
        width, height = 100, 100
        n_wavelengths = 50
        
        # Generate wavelength points
        wavelengths = np.linspace(300, 800, n_wavelengths)
        
        # Create base spectral map
        spectral_map = np.zeros((height, width, 3))  # RGB visualization
        spectral_data = np.zeros((height, width, n_wavelengths))
        
        # Generate realistic spectral patterns
        for i in range(height):
            for j in range(width):
                # Create position-dependent spectrum
                x_rel = i / height
                y_rel = j / width
                
                # Base spectrum with multiple peaks
                spectrum = (
                    100 * np.exp(-(wavelengths - 400)**2 / 1000) +  # Blue peak
                    80 * np.exp(-(wavelengths - 550)**2 / 1000) +   # Green peak
                    60 * np.exp(-(wavelengths - 700)**2 / 1000)     # Red peak
                )
                
                # Add spatial variation
                variation = 0.2 * np.sin(2 * np.pi * x_rel) * np.sin(2 * np.pi * y_rel)
                spectrum *= (1 + variation)
                
                # Add noise
                noise = np.random.normal(0, 2, n_wavelengths)
                final_spectrum = spectrum + noise
                
                # Store spectral data
                spectral_data[i,j] = final_spectrum
                
                # Create RGB visualization
                spectral_map[i,j,0] = np.mean(final_spectrum[wavelengths > 600])  # Red
                spectral_map[i,j,1] = np.mean(final_spectrum[(wavelengths > 500) & (wavelengths < 600)])  # Green
                spectral_map[i,j,2] = np.mean(final_spectrum[wavelengths < 500])  # Blue
        
        # Normalize RGB visualization
        spectral_map = (spectral_map - spectral_map.min()) / (spectral_map.max() - spectral_map.min())
        
        return {
            'spectral_data': spectral_data,
            'wavelengths': wavelengths,
            'visualization': spectral_map,
            'metadata': {
                'dimensions': (height, width),
                'wavelength_range': f"{wavelengths[0]:.0f}-{wavelengths[-1]:.0f}nm",
                'n_wavelengths': n_wavelengths
            },
            'statistics': {
                'mean_intensity': float(np.mean(spectral_data)),
                'max_intensity': float(np.max(spectral_data)),
                'min_intensity': float(np.min(spectral_data)),
                'std_intensity': float(np.std(spectral_data))
            }
        }

    def _render_sample_data_section(self):
        """Render sample data exploration section."""
        st.header("Sample Datasets")
        
        selected_sample = st.selectbox(
            "Select Sample Dataset",
            list(self.sample_datasets.keys())
        )
        
        dataset = self.sample_datasets[selected_sample]
        
        # Display dataset information
        st.subheader("Dataset Information")
        st.write(dataset["description"])
        
        # Visualize data based on type
        if dataset["type"] == "surface_analysis":
            self._visualize_surface_data(dataset["data"])
        elif dataset["type"] == "particle_analysis":
            self._visualize_particle_data(dataset["data"])
        
        # Export options
        if st.button("Use This Dataset"):
            st.session_state.current_data = dataset["data"]
            st.success("Dataset loaded into workspace!")
            
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Download Dataset"):
                self._download_dataset(dataset["data"], selected_sample)
        with col2:
            if st.button("View Full Metadata"):
                st.json(dataset["data"]["metadata"])

    def _render_training_section(self):
        """Render model training interface."""
        st.header("Model Training")
        
        if "current_data" not in st.session_state:
            st.warning("Please load or upload data first!")
            return
            
        # Training parameters
        col1, col2 = st.columns(2)
        with col1:
            model_type = st.selectbox(
                "Model Type",
                ["Surface Analysis", "Particle Detection", "Defect Classification"]
            )
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
            
        with col2:
            advanced_params = st.checkbox("Show Advanced Parameters")
            if advanced_params:
                self._render_advanced_parameters(model_type)
                
        # Training controls
        if st.button("Start Training"):
            with st.spinner("Training in progress..."):
                results = self._train_model(
                    st.session_state.current_data,
                    model_type,
                    test_size
                )
                self._save_training_results(results)
                st.success("Training complete!")
                
                # Display results
                self._display_training_results(results)

    def _save_training_results(self, results: Dict):
        """Save training results and model."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.models_path / f"model_{timestamp}.joblib"
        joblib.dump(results["model"], model_path)
        
        # Save results
        results_path = self.results_path / f"results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump({
                "metrics": results["metrics"],
                "parameters": results["parameters"],
                "timestamp": timestamp
            }, f)
            
        st.session_state.last_training = {
            "model_path": str(model_path),
            "results_path": str(results_path)
        }

    def _render_saved_models(self):
        """Render saved models management interface."""
        st.header("Saved Models")
        
        # List saved models
        saved_models = list(self.models_path.glob("*.joblib"))
        if not saved_models:
            st.info("No saved models found.")
            return
            
        selected_model = st.selectbox(
            "Select Model",
            saved_models,
            format_func=lambda x: f"{x.stem} ({x.stat().st_mtime_ns})"
        )
        
        # Model actions
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Load Model"):
                self._load_model(selected_model)
        with col2:
            if st.button("Export Model"):
                self._export_model(selected_model)
        with col3:
            if st.button("Delete Model"):
                self._delete_model(selected_model)

    def _render_results_section(self):
        """Render analysis results and history."""
        st.header("Analysis Results")
        
        # List saved results
        results_files = list(self.results_path.glob("*.json"))
        if not results_files:
            st.info("No saved results found.")
            return
            
        selected_result = st.selectbox(
            "Select Result",
            results_files,
            format_func=lambda x: f"{x.stem} ({x.stat().st_mtime_ns})"
        )
        
        # Display results
        with open(selected_result) as f:
            results = json.load(f)
            
        # Visualize results
        st.subheader("Performance Metrics")
        self._visualize_results(results)
        
        # Export options
        if st.button("Export Results"):
            self._export_results(results)

    def _visualize_surface_data(self, data: Dict):
        """Create 3D surface plot of sample data."""
        fig = px.scatter_3d(
            x=data["topology"]["x"],
            y=data["topology"]["y"],
            z=data["topology"]["z"],
            color=data["topology"]["z"],
            title="Surface Topology"
        )
        st.plotly_chart(fig)
        
        # Spectral visualization
        st.subheader("Spectral Data")
        fig = px.line(
            x=np.arange(data["spectral"].shape[1]),
            y=data["spectral"].mean(axis=0),
            title="Average Spectral Response"
        )
        st.plotly_chart(fig)

    def _download_dataset(self, data: Dict, name: str):
        """Create downloadable version of dataset."""
        # Convert to pandas DataFrame
        df_topology = pd.DataFrame(data["topology"])
        df_spectral = pd.DataFrame(data["spectral"])
        
        # Create Excel writer
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_topology.to_excel(writer, sheet_name='Topology')
            df_spectral.to_excel(writer, sheet_name='Spectral')
            pd.DataFrame([data["metadata"]]).to_excel(writer, sheet_name='Metadata')
            
        # Download button
        st.download_button(
            label="Download Excel",
            data=output.getvalue(),
            file_name=f"{name.lower().replace(' ', '_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    def render_interface(self):
        """Render model management interface."""
        st.title("Model Management")
        
        tab1, tab2, tab3 = st.tabs([
            "Model Training",
            "Model Evaluation",
            "Model Deployment"
        ])
        
        with tab1:
            self._render_training_interface()
        
        with tab2:
            self._render_evaluation_interface()
        
        with tab3:
            self._render_deployment_interface()
    
    def _render_training_interface(self):
        """Render model training interface."""
        st.subheader("Train New Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Model Type",
                ["CNN", "Transformer", "Hybrid"]
            )
            
            epochs = st.slider("Training Epochs", 10, 100, 50)
            
        with col2:
            batch_size = st.selectbox(
                "Batch Size",
                [16, 32, 64, 128]
            )
            
            learning_rate = st.select_slider(
                "Learning Rate",
                options=[0.1, 0.01, 0.001, 0.0001]
            )
        
        if st.button("Start Training"):
            self._simulate_training(model_type, epochs, batch_size, learning_rate)
    
    def _render_evaluation_interface(self):
        """Render model evaluation interface."""
        st.subheader("Model Evaluation")
        
        if self.models:
            model_name = st.selectbox(
                "Select Model",
                list(self.models.keys())
            )
            
            metrics = self._simulate_evaluation(model_name)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            with col2:
                st.metric("Precision", f"{metrics['precision']:.2%}")
            with col3:
                st.metric("Recall", f"{metrics['recall']:.2%}")
            
            # Plot confusion matrix
            self._plot_confusion_matrix(metrics['confusion_matrix'])
        
        else:
            st.info("No trained models available. Train a model first.")
    
    def _render_deployment_interface(self):
        """Render model deployment interface."""
        st.subheader("Model Deployment")
        
        if self.models:
            model_name = st.selectbox(
                "Select Model to Deploy",
                list(self.models.keys())
            )
            
            deployment_type = st.radio(
                "Deployment Type",
                ["Local", "Cloud", "Edge"]
            )
            
            if st.button("Deploy Model"):
                with st.spinner("Deploying model..."):
                    st.success(f"Model {model_name} deployed successfully!")
        else:
            st.info("No models available for deployment.")
    
    def _simulate_training(self, model_type: str, epochs: int, 
                         batch_size: int, learning_rate: float):
        """Simulate model training."""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Simulate training progress
        for i in range(epochs):
            progress = (i + 1) / epochs
            progress_bar.progress(progress)
            status_text.text(f"Training progress: {progress:.0%}")
            
            # Simulate metrics
            train_loss = 1.0 - progress + 0.1 * np.random.randn()
            val_loss = 1.1 - progress + 0.1 * np.random.randn()
            
            # Store training history
            if model_type not in self.training_history:
                self.training_history[model_type] = {
                    'train_loss': [], 'val_loss': []
                }
            
            self.training_history[model_type]['train_loss'].append(train_loss)
            self.training_history[model_type]['val_loss'].append(val_loss)
        
        # Plot training history
        self._plot_training_history(model_type)
        
        # Store model configuration
        self.model_configs[model_type] = {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
        
        st.success(f"{model_type} model trained successfully!")
    
    def _plot_training_history(self, model_type: str):
        """Plot training history."""
        history = self.training_history[model_type]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            y=history['train_loss'],
            name='Training Loss',
            mode='lines'
        ))
        
        fig.add_trace(go.Scatter(
            y=history['val_loss'],
            name='Validation Loss',
            mode='lines'
        ))
        
        fig.update_layout(
            title=f"{model_type} Training History",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _simulate_evaluation(self, model_name: str) -> Dict[str, Any]:
        """Simulate model evaluation."""
        return {
            'accuracy': np.random.uniform(0.85, 0.95),
            'precision': np.random.uniform(0.80, 0.90),
            'recall': np.random.uniform(0.80, 0.90),
            'confusion_matrix': np.random.randint(0, 100, size=(4, 4))
        }
    
    def _plot_confusion_matrix(self, confusion_matrix: np.ndarray):
        """Plot confusion matrix."""
        fig = go.Figure(data=[
            go.Heatmap(
                z=confusion_matrix,
                colorscale='Viridis'
            )
        ])
        
        fig.update_layout(
            title="Confusion Matrix",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True) 