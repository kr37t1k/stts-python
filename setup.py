from setuptools import setup, find_packages

setup(
    name="stts-py3",
    version="0.2.0",
    description="Python package for using Silero TTS",
    author="kr37t1k",
    author_email="egorakentiev28@gmail.com",
    packages=find_packages(),
    install_requires=[
        "requirements.txt"
    ],
    python_requires=">=3.7",
)
