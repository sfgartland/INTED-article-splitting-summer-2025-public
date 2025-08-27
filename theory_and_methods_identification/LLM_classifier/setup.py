"""
Setup script for the LLM_classifier module.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="section-classifier",
    version="0.1.0",
    author="INTED Article Splitting Project",
    description="A modular Python system for classifying academic sections using OpenAI's GPT models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/section-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.7",
    install_requires=[
        "pandas>=1.3.0",
        "openai>=1.0.0",
        "pydantic>=1.8.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    keywords="nlp, classification, academic, research, openai, gpt",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/section-classifier/issues",
        "Source": "https://github.com/your-repo/section-classifier",
        "Documentation": "https://github.com/your-repo/section-classifier#readme",
    },
) 