import pytest
from src.ecs_sehi_analyzer.core.research.ecs_analysis import CarbonMaterial, ECSAnalyzer

@pytest.fixture
def analyzer():
    return ECSAnalyzer()

@pytest.fixture
def sample_material():
    return CarbonMaterial(
        material_id="TEST-001",
        graphene_content=85.0,
        surface_area=1200.0,
        degradation_rate=0.15
    )

def test_add_material(analyzer, sample_material):
    result = analyzer.add_material(sample_material)
    assert result["status"] == "success"
    assert result["material_id"] == "TEST-001"
    assert len(analyzer.materials_db) == 1

def test_predict_degradation(analyzer, sample_material):
    analyzer.add_material(sample_material)
    prediction = analyzer.predict_degradation("TEST-001")
    assert "predicted_lifetime" in prediction
    assert "stability_index" in prediction
    assert prediction["confidence"] > 0.9

def test_analyze_surface_properties(analyzer, sample_material):
    analyzer.add_material(sample_material)
    analysis = analyzer.analyze_surface_properties("TEST-001")
    assert "surface_area" in analysis
    assert "graphene_quality" in analysis
    assert "defect_density" in analysis
    assert len(analysis["recommendations"]) > 0 