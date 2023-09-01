"""Setup file."""

from setuptools import setup, find_packages

setup(
    name="rl_benchmarks",
    version="0.1.0",
    description="Scaling Self-Supervised Learning for Histopathology with Masked Image Modeling",
    author="Owkin, Inc., New York, NY, USA",
    url="https://github.com/owkin/HistoSSLscaling",
    python_requires="==3.8.*",
    packages=find_packages(),
    long_description=open("README.md", encoding="latin-1").read(),
)
