from setuptools import setup, find_packages

setup(
    name="divide21x",
    version="0.1.0",
    author="Jacinto Jeje Matamba Quimua",
    description="Divide21X Phase 1: Action-only benchmark environment for Divide21.",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29",
        "divide21env>=0.1.3",
    ],
    python_requires=">=3.10",
)
