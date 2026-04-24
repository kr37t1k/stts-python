from setuptools import setup, find_packages

setup(
    name="silerotts",
    version="0.8",
    description="Python package for using Silero TTS",
    author="daswer123",
    author_email="",
    packages=find_packages(),
    include_dirs=["stts", "tests"],
    install_requires=[
        "torch==2.3.0",
        "pyyaml>=6.0.1",
        "torchaudio",
        "pytest",
        "numpy<2",
    ],
    python_requires=">=3.7",
)
