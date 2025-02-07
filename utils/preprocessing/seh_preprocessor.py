import numpy as np

class SEHIPreprocessor:
    def normalize(self, data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))
    
    def remove_noise(self, data, threshold=0.8):
        return np.where(data < threshold, 0, data)
