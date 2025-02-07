import numpy as np
from scipy import fft

class ChronoAnalyzer:
    def analyze_time_series(self, data: np.ndarray) -> dict:
        fft_values = fft.fft(data)
        frequencies = fft.fftfreq(len(data))
        return {
            "dominant_frequency": frequencies[np.argmax(np.abs(fft_values))],
            "spectral_entropy": self._calculate_entropy(np.abs(fft_values))
        }

    def _calculate_entropy(self, spectrum: np.ndarray) -> float:
        normalized = spectrum / np.sum(spectrum)
        return -np.sum(normalized * np.log2(normalized))
