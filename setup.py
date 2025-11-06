from setuptools import setup, find_packages

setup(
    name="divide21x",
    version="0.1.0",
    author="Jacinto Jeje Matamba Quimua",
    description="Divide21X Phase 1: Action-State benchmark environment for faithful strategic reasoning.",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29",
        "divide21env>=0.2.1",
    ],
    python_requires=">=3.10",
)
