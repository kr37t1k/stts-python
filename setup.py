from setuptools import setup, find_packages

setup(
    name="silerotts",
    version="0.8.0",
    description="Python Server Silero TTS with Analytics Dashboard",
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
        "fastapi>=0.109.0",
        "uvicorn>=0.27.0",
        "jinja2>=3.1.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
    ],
    python_requires=">=3.8",
    extras_require={
        "dev": [
            "pytest",
            "pytest-asyncio",
            "black",
            "flake8",
        ],
        "analytics": [
            "psutil>=5.9.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    keywords="text-to-speech tts silero audio speech synthesis api server",
    url="https://github.com/kr37t1k/stts-python",
)
