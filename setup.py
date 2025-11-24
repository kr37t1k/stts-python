from setuptools import setup, find_packages

setup(
    name="silerotts",
    version="0.4.1",
    description="Python package for using Silero TTS",
    author="kr37t1k",
    author_email="egorakentiev28@gmail.com",
    packages=find_packages(),
    include_dirs=["stts", "tests"],
    install_requires=[
        "torch==2.3.0",
        "pyyaml>=6.0.1",
        "torchaudio",
        "pytest",
        "numpy",
    ],
    python_requires=">=3.7",
)
