from setuptools import setup, find_packages

setup(
    name="silerotts",
    version="0.7.1",
    description="Python Server Silero TTS",
    author="daswer123",
    author_email="",
    packages=find_packages(),
    include_dirs=["stts", "tests"],
    package_data={
        "stts": ["templates/*.html", "static/*.*"]
    },
    include_package_data=True, 
    install_requires=[
        "torch==2.3.0",
        "pyyaml>=6.0.1",
        "torchaudio",
        "pytest",
        "numpy<2",
    ],
    python_requires=">=3.7",
)
