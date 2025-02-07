import unittest
import numpy as np
from src.ecs_sehi_analyzer.core.analysis_engine import SEHIAnalysisEngine, AnalysisConfig

class TestAnalysisEngine(unittest.TestCase):
    def setUp(self):
        self.config = AnalysisConfig(resolution=512, noise_reduction=0.8)
        self.engine = SEHIAnalysisEngine(self.config)
        self.sample_data = np.random.normal(0, 1, (512, 512))

    def test_basic_analysis(self):
        results = self.engine.analyze_chemical_distribution(self.sample_data)
        self.assertIn('mean', results)
        self.assertAlmostEqual(results['mean'], np.mean(self.sample_data), delta=0.01)

    def test_phase_detection(self):
        data = np.random.rand(512, 512)
        boundaries, metrics = self.engine.detect_phase_boundaries(data)
        self.assertEqual(boundaries.shape, data.shape)
        self.assertIn('max_gradient', metrics)
