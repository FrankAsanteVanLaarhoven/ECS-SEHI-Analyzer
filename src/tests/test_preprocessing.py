import pytest
import numpy as np
from app.utils.preprocessing import DataLoader
from pathlib import Path

class TestDataLoader:
    def test_load_local_data(self, tmp_path):
        """Test loading local data files."""
        # Create test data
        test_data = np.random.rand(10, 10)
        test_file = tmp_path / "test.csv"
        np.savetxt(test_file, test_data, delimiter=",")
        
        loader = DataLoader()
        result = loader.load_local_data(test_file)
        
        assert result is not None
        assert result.shape == (10, 10)
        
    def test_invalid_format(self):
        """Test handling of invalid file formats."""
        loader = DataLoader()
        with pytest.raises(ValueError):
            loader.load_local_data("invalid.xyz")
            
    def test_api_data_loading(self, requests_mock):
        """Test API data loading."""
        mock_data = {'data': [1, 2, 3]}
        requests_mock.get('http://api.test.com', json=mock_data)
        
        loader = DataLoader()
        result = loader.load_api_data('http://api.test.com')
        
        assert result is not None
        assert len(result) == 3 