import pytest
import sys
from pathlib import Path

def main():
    """Run all tests with coverage report."""
    test_dir = Path(__file__).parent
    
    # Run tests with coverage
    exit_code = pytest.main([
        str(test_dir),
        '--cov=app',
        '--cov=models',
        '--cov-report=term-missing',
        '--cov-report=html',
        '-v'
    ])
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main() 