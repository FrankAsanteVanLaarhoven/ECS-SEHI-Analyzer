from setuptools import setup, find_packages

setup(
    name="ecs-sehi-analyzer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "streamlit>=1.28.0",
        "streamlit-ace>=0.1.1",
        "pygments>=2.15.1",
        "plotly>=5.18.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
        ],
    },
    python_requires=">=3.8",
)
