from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
import streamlit as st
from ..quantum.entanglement import QuantumEntanglementEngine
from ..quantum.error_correction import QuantumErrorCorrector
from ..neural.material_network import MaterialAnalysisNetwork, NetworkConfig

@dataclass
class SyncConfig:
    """Neural synchronization configuration"""
    sync_interval: float = 1.0  # seconds
    batch_size: int = 32
    confidence_threshold: float = 0.95
    max_retries: int = 3
    quantum_noise_threshold: float = 0.1
    metadata: Dict = field(default_factory=dict)

class NeuroSyncEngine:
    def __init__(self, config: Optional[SyncConfig] = None):
        self.config = config or SyncConfig()
        self.entanglement = QuantumEntanglementEngine()
        self.error_corrector = QuantumErrorCorrector()
        self.sync_history: List[Dict] = []
        self.active_models: Dict[str, MaterialAnalysisNetwork] = {}
        
    def register_model(self, model_id: str, model: MaterialAnalysisNetwork):
        """Register neural network for synchronization"""
        self.active_models[model_id] = model
        
    def sync_parameters(self, source_id: str, target_id: str) -> Dict:
        """Synchronize neural network parameters"""
        if source_id not in self.active_models or target_id not in self.active_models:
            raise ValueError("Invalid model IDs")
            
        source_model = self.active_models[source_id]
        target_model = self.active_models[target_id]
        
        try:
            # Create quantum channel
            channel = self.entanglement.secure_channel(target_id)
            
            # Serialize and sync parameters
            sync_result = self._sync_model_parameters(
                source_model, 
                target_model,
                channel
            )
            
            # Record sync event
            self._record_sync_event(source_id, target_id, sync_result)
            
            return {
                "status": "success",
                "synced_parameters": sync_result["synced_params"],
                "confidence": sync_result["confidence"]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
            
    def analyze_sync_quality(self) -> Dict:
        """Analyze synchronization quality"""
        if not self.sync_history:
            return {}
            
        total_syncs = len(self.sync_history)
        successful_syncs = sum(
            1 for sync in self.sync_history 
            if sync["confidence"] >= self.config.confidence_threshold
        )
        
        analysis = {
            "total_syncs": total_syncs,
            "success_rate": successful_syncs / total_syncs,
            "average_confidence": np.mean([s["confidence"] for s in self.sync_history]),
            "error_rate": self._calculate_error_rate(),
            "sync_trends": self._analyze_sync_trends()
        }
        
        return analysis
        
    def render_sync_dashboard(self):
        """Render Streamlit sync dashboard"""
        st.markdown("### 🧠 Quantum Neural Sync")
        
        # Active models
        st.markdown("#### Active Models")
        for model_id, model in self.active_models.items():
            with st.expander(f"Model {model_id}"):
                st.code(str(model))
                
        # Sync metrics
        if self.sync_history:
            analysis = self.analyze_sync_quality()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Success Rate",
                    f"{analysis['success_rate']:.1%}"
                )
            with col2:
                st.metric(
                    "Avg Confidence",
                    f"{analysis['average_confidence']:.2f}"
                )
            with col3:
                st.metric(
                    "Error Rate",
                    f"{analysis['error_rate']:.2%}"
                )
                
            # Sync trends chart
            st.markdown("#### Sync Trends")
            trends = analysis["sync_trends"]
            
            import plotly.graph_objects as go
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=[t["timestamp"] for t in self.sync_history],
                y=[t["confidence"] for t in self.sync_history],
                name="Confidence",
                mode="lines+markers"
            ))
            
            fig.update_layout(
                title="Sync Confidence Over Time",
                xaxis_title="Time",
                yaxis_title="Confidence",
                yaxis_range=[0, 1]
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
    def _sync_model_parameters(self, 
                             source_model: MaterialAnalysisNetwork,
                             target_model: MaterialAnalysisNetwork,
                             channel) -> Dict:
        """Synchronize model parameters through quantum channel"""
        sync_results = {
            "synced_params": 0,
            "total_params": 0,
            "confidence": 0.0
        }
        
        for source_param, target_param in zip(
            source_model.parameters(), 
            target_model.parameters()
        ):
            # Convert parameters to bytes
            param_data = source_param.data.cpu().numpy().tobytes()
            
            # Apply error correction
            errors = self.error_corrector.detect_errors(param_data)
            corrected_data = self.error_corrector.correct_errors(param_data, errors)
            
            # Send through quantum channel
            transmitted_data = channel.send({"param_data": corrected_data})
            
            # Update target parameters
            received_data = channel.receive(transmitted_data)
            param_array = np.frombuffer(
                received_data["param_data"], 
                dtype=source_param.data.dtype
            ).reshape(source_param.data.shape)
            
            target_param.data.copy_(torch.tensor(param_array))
            
            sync_results["synced_params"] += 1
            sync_results["total_params"] += param_array.size
            
        # Calculate confidence based on error correction
        sync_results["confidence"] = 1.0 - (
            len(errors) / sync_results["total_params"]
            if sync_results["total_params"] > 0 else 0
        )
        
        return sync_results
        
    def _record_sync_event(self, source_id: str, target_id: str, result: Dict):
        """Record synchronization event"""
        self.sync_history.append({
            "source": source_id,
            "target": target_id,
            "timestamp": datetime.now(),
            "confidence": result["confidence"],
            "synced_params": result["synced_params"],
            "total_params": result["total_params"]
        })
        
    def _calculate_error_rate(self) -> float:
        """Calculate synchronization error rate"""
        if not self.sync_history:
            return 0.0
            
        return 1.0 - np.mean([
            sync["confidence"]
            for sync in self.sync_history
        ])
        
    def _analyze_sync_trends(self) -> Dict:
        """Analyze synchronization trends"""
        if len(self.sync_history) < 2:
            return {}
            
        confidences = [sync["confidence"] for sync in self.sync_history]
        trend = np.polyfit(range(len(confidences)), confidences, 1)[0]
        
        return {
            "confidence_trend": trend,
            "trend_direction": "improving" if trend > 0 else "degrading"
        } 