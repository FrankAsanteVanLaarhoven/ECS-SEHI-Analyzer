# tests/benchmark.py
import timeit
import numpy as np

setup = '''
from src.ecs_sehi_analyzer.core.analysis_engine import SEHIAnalysisEngine
data = np.random.rand(2048, 2048)
engine = SEHIAnalysisEngine()
'''

print("Analysis Benchmark:", timeit.timeit('engine.analyze_chemical_distribution(data)', 
                                          setup=setup, number=100))
