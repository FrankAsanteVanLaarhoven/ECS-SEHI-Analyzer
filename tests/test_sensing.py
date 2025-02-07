import pytest
import numpy as np
from datetime import datetime
from src.ecs_sehi_analyzer.core.quantum.sensing import (
    QuantumSensor,
    SensorConfig,
    SensorType,
    SensorMode
)

@pytest.fixture
def sensor():
    return QuantumSensor()

@pytest.fixture
def sample_location():
    return (0.0, 0.0, 0.0)

def test_sensor_initialization(sensor, sample_location):
    result = sensor.initialize_sensor(
        "sensor_1",
        SensorType.MAGNETIC,
        sample_location
    )
    
    assert result["success"]
    assert "sensor_1" in sensor.sensors
    assert sensor.sensors["sensor_1"]["type"] == SensorType.MAGNETIC
    assert sensor.sensors["sensor_1"]["location"] == sample_location
    assert "calibration" in result

def test_duplicate_sensor(sensor, sample_location):
    sensor.initialize_sensor("sensor_1", SensorType.MAGNETIC, sample_location)
    
    with pytest.raises(ValueError):
        sensor.initialize_sensor("sensor_1", SensorType.MAGNETIC, sample_location)

def test_single_measurement(sensor, sample_location):
    sensor.initialize_sensor("sensor_1", SensorType.MAGNETIC, sample_location)
    
    result = sensor.measure("sensor_1", SensorMode.SINGLE)
    
    assert result["success"]
    assert "measurement_id" in result
    assert "value" in result["result"]
    assert "uncertainty" in result["result"]
    assert "raw_counts" in result["result"]

def test_distributed_measurement(sensor, sample_location):
    # Initialize multiple sensors
    sensors = ["sensor_1", "sensor_2", "sensor_3"]
    for i, sid in enumerate(sensors):
        location = (float(i), 0.0, 0.0)
        sensor.initialize_sensor(sid, SensorType.MAGNETIC, location)
    
    result = sensor.measure(sensors, SensorMode.DISTRIBUTED)
    
    assert result["success"]
    assert "value" in result["result"]
    assert "uncertainty" in result["result"]
    assert "individual_results" in result["result"]
    assert len(result["result"]["individual_results"]) == len(sensors)

def test_entangled_measurement(sensor, sample_location):
    # Initialize pair of sensors
    sensor.initialize_sensor("sensor_1", SensorType.MAGNETIC, (0.0, 0.0, 0.0))
    sensor.initialize_sensor("sensor_2", SensorType.MAGNETIC, (1.0, 0.0, 0.0))
    
    result = sensor.measure(["sensor_1", "sensor_2"], SensorMode.ENTANGLED)
    
    assert result["success"]
    assert "value" in result["result"]
    assert "uncertainty" in result["result"]
    assert "enhancement_factor" in result["result"]
    assert result["result"]["enhancement_factor"] == np.sqrt(2)

def test_adaptive_measurement(sensor, sample_location):
    sensor.initialize_sensor("sensor_1", SensorType.MAGNETIC, sample_location)
    
    result = sensor.measure("sensor_1", SensorMode.ADAPTIVE)
    
    assert result["success"]
    assert "value" in result["result"]
    assert "uncertainty" in result["result"]
    assert "initial_value" in result["result"]
    assert "adaptation_phase" in result["result"]

def test_data_analysis(sensor, sample_location):
    # Create some measurements
    sensor.initialize_sensor("sensor_1", SensorType.MAGNETIC, sample_location)
    for _ in range(3):
        sensor.measure("sensor_1", SensorMode.SINGLE)
    
    analysis = sensor.analyze_data()
    
    assert analysis["success"]
    assert "mean" in analysis["analysis"]
    assert "std" in analysis["analysis"]
    assert "uncertainty" in analysis["analysis"]
    assert analysis["analysis"]["num_measurements"] == 3

def test_calibration(sensor, sample_location):
    sensor.initialize_sensor("sensor_1", SensorType.MAGNETIC, sample_location)
    
    result = sensor._calibrate_sensor("sensor_1")
    
    assert result["success"]
    assert "baseline" in result
    assert "timestamp" in result
    assert sensor.sensors["sensor_1"]["last_calibration"] is not None
    assert "sensor_1" in sensor.calibration_data

def test_invalid_sensor_measurement(sensor):
    result = sensor.measure("invalid_sensor", SensorMode.SINGLE)
    assert not result["success"]
    assert "error" in result

def test_sensor_config():
    config = SensorConfig(
        num_sensors=8,
        precision=1e-12,
        sampling_rate=2000
    )
    sensor = QuantumSensor(config)
    
    assert sensor.config.num_sensors == 8
    assert sensor.config.precision == 1e-12
    assert sensor.config.sampling_rate == 2000

def test_measurement_history(sensor, sample_location):
    sensor.initialize_sensor("sensor_1", SensorType.MAGNETIC, sample_location)
    
    # Perform multiple measurements
    for mode in SensorMode:
        sensor.measure("sensor_1", mode)
    
    assert len(sensor.measurements) == len(SensorMode)
    
    for measurement in sensor.measurements:
        assert "timestamp" in measurement
        assert "sensors" in measurement
        assert "mode" in measurement
        assert "duration" in measurement
        assert "result" in measurement 