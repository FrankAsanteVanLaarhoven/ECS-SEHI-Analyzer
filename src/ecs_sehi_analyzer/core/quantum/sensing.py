from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from enum import Enum
import streamlit as st
from datetime import datetime
from .circuit import QuantumCircuitEngine, GateType
from .entanglement import QuantumEntanglementEngine, EntanglementType
from .noise import NoiseSimulator

class SensorType(Enum):
    MAGNETIC = "magnetic"
    ELECTRIC = "electric"
    GRAVITATIONAL = "gravitational"
    CUSTOM = "custom"

class SensorMode(Enum):
    SINGLE = "single"
    DISTRIBUTED = "distributed"
    ENTANGLED = "entangled"
    ADAPTIVE = "adaptive"

@dataclass
class SensorConfig:
    """Quantum sensor configuration"""
    num_sensors: int = 4
    precision: float = 1e-9  # Base precision
    sampling_rate: float = 1000  # Hz
    integration_time: float = 1.0  # seconds
    noise_threshold: float = 0.01
    metadata: Dict = field(default_factory=dict)

class QuantumSensor:
    def __init__(self, config: Optional[SensorConfig] = None):
        self.config = config or SensorConfig()
        self.circuit = QuantumCircuitEngine()
        self.entanglement = QuantumEntanglementEngine()
        self.noise = NoiseSimulator()
        
        self.sensors: Dict[str, Dict] = {}
        self.measurements: List[Dict] = []
        self.calibration_data: Dict[str, np.ndarray] = {}
        
    def initialize_sensor(self, 
                        sensor_id: str,
                        sensor_type: SensorType,
                        location: Tuple[float, float, float]) -> Dict:
        """Initialize quantum sensor"""
        if sensor_id in self.sensors:
            raise ValueError(f"Sensor {sensor_id} already exists")
            
        self.sensors[sensor_id] = {
            "type": sensor_type,
            "location": location,
            "status": "initialized",
            "last_calibration": None,
            "metadata": {}
        }
        
        # Perform initial calibration
        calibration = self._calibrate_sensor(sensor_id)
        
        return {
            "success": True,
            "sensor_id": sensor_id,
            "calibration": calibration
        }
        
    def measure(self, 
               sensor_ids: Union[str, List[str]],
               mode: SensorMode = SensorMode.SINGLE,
               duration: float = 1.0) -> Dict:
        """Perform quantum sensing measurement"""
        try:
            if isinstance(sensor_ids, str):
                sensor_ids = [sensor_ids]
                
            # Verify sensors
            for sid in sensor_ids:
                if sid not in self.sensors:
                    raise ValueError(f"Unknown sensor: {sid}")
                    
            # Initialize measurement circuit
            self.circuit.initialize_circuit()
            
            # Setup based on mode
            if mode == SensorMode.SINGLE:
                result = self._single_sensor_measurement(sensor_ids[0], duration)
            elif mode == SensorMode.DISTRIBUTED:
                result = self._distributed_measurement(sensor_ids, duration)
            elif mode == SensorMode.ENTANGLED:
                result = self._entangled_measurement(sensor_ids, duration)
            elif mode == SensorMode.ADAPTIVE:
                result = self._adaptive_measurement(sensor_ids, duration)
            else:
                raise ValueError(f"Unknown sensor mode: {mode}")
                
            # Record measurement
            measurement = {
                "timestamp": datetime.now(),
                "sensors": sensor_ids,
                "mode": mode.value,
                "duration": duration,
                "result": result
            }
            self.measurements.append(measurement)
            
            return {
                "success": True,
                "measurement_id": len(self.measurements) - 1,
                "result": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def analyze_data(self, measurement_ids: Optional[List[int]] = None) -> Dict:
        """Analyze quantum sensing data"""
        if measurement_ids is None:
            measurement_ids = range(len(self.measurements))
            
        try:
            measurements = [self.measurements[i] for i in measurement_ids]
            
            # Calculate statistics
            values = [m["result"]["value"] for m in measurements]
            uncertainties = [m["result"]["uncertainty"] for m in measurements]
            
            analysis = {
                "mean": np.mean(values),
                "std": np.std(values),
                "uncertainty": np.sqrt(np.sum(np.array(uncertainties)**2))/len(uncertainties),
                "num_measurements": len(measurements)
            }
            
            return {
                "success": True,
                "analysis": analysis
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def render_sensing_interface(self):
        """Render Streamlit sensing interface"""
        st.markdown("### 📡 Quantum Sensing")
        
        # Sensor management
        st.markdown("#### Sensors")
        for sensor_id, info in self.sensors.items():
            st.markdown(f"""
            - **{sensor_id}** ({info['type'].value})
              - Location: {info['location']}
              - Status: {info['status']}
              - Last Calibration: {info['last_calibration']}
            """)
            
        # New measurement
        st.markdown("#### New Measurement")
        selected_sensors = st.multiselect(
            "Select Sensors",
            list(self.sensors.keys())
        )
        
        mode = st.selectbox(
            "Measurement Mode",
            [m.value for m in SensorMode]
        )
        
        duration = st.slider(
            "Duration (s)",
            0.1, 10.0, 1.0
        )
        
        if st.button("Start Measurement"):
            result = self.measure(
                selected_sensors,
                SensorMode(mode),
                duration
            )
            
            if result["success"]:
                st.success("Measurement completed!")
                st.json(result["result"])
            else:
                st.error(f"Measurement failed: {result.get('error')}")
                
        # Measurement history
        if self.measurements:
            st.markdown("#### Measurement History")
            self._render_measurement_history()
            
    def _calibrate_sensor(self, sensor_id: str) -> Dict:
        """Calibrate quantum sensor"""
        try:
            # Perform calibration sequence
            self.circuit.initialize_circuit()
            
            # Add calibration operations
            self.circuit.add_gate(GateType.HADAMARD, [0])
            self.circuit.add_gate(GateType.PHASE, [0], [np.pi/4])
            
            # Execute and analyze
            result = self.circuit.execute_circuit()
            
            # Store calibration data
            self.calibration_data[sensor_id] = np.array(result["quantum_state"])
            self.sensors[sensor_id]["last_calibration"] = datetime.now()
            
            return {
                "success": True,
                "baseline": np.mean(result["counts"].values()),
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def _single_sensor_measurement(self, sensor_id: str, duration: float) -> Dict:
        """Perform single sensor measurement"""
        # Create sensing circuit
        self.circuit.add_gate(GateType.HADAMARD, [0])
        
        # Simulate interaction with measured field
        phase = np.random.normal(0, self.config.noise_threshold)
        self.circuit.add_gate(GateType.PHASE, [0], [phase])
        
        # Measure
        result = self.circuit.execute_circuit()
        
        # Calculate measured value and uncertainty
        counts = result["counts"]
        total_shots = sum(counts.values())
        p0 = counts.get("0", 0) / total_shots
        
        value = np.arccos(2*p0 - 1)
        uncertainty = self.config.precision / np.sqrt(total_shots)
        
        return {
            "value": value,
            "uncertainty": uncertainty,
            "raw_counts": counts
        }
        
    def _distributed_measurement(self, sensor_ids: List[str], duration: float) -> Dict:
        """Perform distributed sensing measurement"""
        # Initialize all sensors
        values = []
        uncertainties = []
        
        for sid in sensor_ids:
            result = self._single_sensor_measurement(sid, duration)
            values.append(result["value"])
            uncertainties.append(result["uncertainty"])
            
        # Combine results
        combined_value = np.mean(values)
        combined_uncertainty = np.sqrt(np.sum(np.array(uncertainties)**2))/len(uncertainties)
        
        return {
            "value": combined_value,
            "uncertainty": combined_uncertainty,
            "individual_results": dict(zip(sensor_ids, values))
        }
        
    def _entangled_measurement(self, sensor_ids: List[str], duration: float) -> Dict:
        """Perform entangled sensing measurement"""
        # Create GHZ state between sensors
        self.entanglement.create_entanglement(
            [(i, i+1) for i in range(len(sensor_ids)-1)],
            EntanglementType.GHZ
        )
        
        # Add sensing operations
        for i in range(len(sensor_ids)):
            phase = np.random.normal(0, self.config.noise_threshold)
            self.circuit.add_gate(GateType.PHASE, [i], [phase])
            
        # Measure
        result = self.circuit.execute_circuit()
        
        # Enhanced precision due to entanglement
        enhancement_factor = np.sqrt(len(sensor_ids))
        value = np.mean(list(result["counts"].values()))
        uncertainty = self.config.precision / enhancement_factor
        
        return {
            "value": value,
            "uncertainty": uncertainty,
            "enhancement_factor": enhancement_factor
        }
        
    def _adaptive_measurement(self, sensor_ids: List[str], duration: float) -> Dict:
        """Perform adaptive sensing measurement"""
        # Initial measurement
        result = self._distributed_measurement(sensor_ids, duration/2)
        initial_value = result["value"]
        
        # Adjust measurement basis based on initial result
        phase_adjustment = -initial_value
        
        # Second measurement with adjusted basis
        for sid in sensor_ids:
            self.circuit.add_gate(GateType.PHASE, [0], [phase_adjustment])
            
        final_result = self._distributed_measurement(sensor_ids, duration/2)
        
        return {
            "value": final_result["value"],
            "uncertainty": final_result["uncertainty"] * 0.7,  # Reduced uncertainty due to adaptation
            "initial_value": initial_value,
            "adaptation_phase": phase_adjustment
        }
        
    def _render_measurement_history(self):
        """Render measurement history visualization"""
        import plotly.graph_objects as go
        
        # Create measurement history plot
        timestamps = [m["timestamp"] for m in self.measurements]
        values = [m["result"]["value"] for m in self.measurements]
        uncertainties = [m["result"]["uncertainty"] for m in self.measurements]
        
        fig = go.Figure()
        
        # Add value trace
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=values,
            mode='lines+markers',
            name='Measured Value',
            error_y=dict(
                type='data',
                array=uncertainties,
                visible=True
            )
        ))
        
        fig.update_layout(
            title="Measurement History",
            xaxis_title="Time",
            yaxis_title="Value",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True) 