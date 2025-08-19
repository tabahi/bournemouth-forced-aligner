# setup.py - Final Production Version
from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            return f.read()
    return "Bournemouth Forced Aligner - Phoneme-level timestamp extraction"

setup(
    name="bournemouth-forced-aligner",
    version="0.1.0",
    packages=find_packages(exclude=["tests*", "docs*", "examples*"]),
    python_requires=">=3.8",
    
    # Core dependencies
    install_requires=[
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
        "huggingface_hub>=0.8.0",
        "numpy>=1.19.0",
        "click>=8.0.0",
        "phonemizer>=3.3.0",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "test": [
            "pytest>=6.0",
            "pytest-cov>=3.0.0",
            "pytest-xdist>=2.5.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
        "audio": [
            "librosa>=0.8.0",
            "soundfile>=0.10.0",
            "pydub>=0.25.0",
        ],
    },
    
    # CLI entry point
    entry_points={
        "console_scripts": [
            "balign=bournemouth_aligner.cli:main",
        ],
    },
    
    # Package metadata
    author="Tabahi",
    author_email="tabahi@duck.com",
    description="Bournemouth Forced Aligner - Phoneme-level timestamp extraction",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/tabahi/bournemouth-forced-aligner",
    project_urls={
        "Documentation": "https://github.com/tabahi/bournemouth-forced-aligner#readme",
        "Bug Tracker": "https://github.com/tabahi/bournemouth-forced-aligner/issues",
        "Source Code": "https://github.com/tabahi/bournemouth-forced-aligner",
        "CUPE Models": "https://huggingface.co/Tabahi/CUPE-2i",
    },
    
    # Package classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "License :: OSI Approved :: GPL V3",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
    
    # Keywords for discovery
    keywords=[
        "phoneme", "alignment", "speech", "audio", "timestamp", 
        "forced-alignment", "bournemouth", "CUPE", "speech-recognition", "linguistics"
    ],
    
    # Package data
    package_data={
        "bournemouth_aligner": ["py.typed", "*.txt", "*.md"],
    },
    include_package_data=True,
    
    # Minimum Python version
    zip_safe=False,
)