#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReID Embedding Extractor 설치 스크립트
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="reid-embedding-extractor",
    version="1.0.0",
    author="ReID Team",
    author_email="reid@example.com",
    description="SOLIDER_REID 모델을 사용한 ReID 임베딩 추출 도구",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/reid-embedding-extractor",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
        ],
        "gpu": [
            "faiss-gpu>=1.7.0",
        ],
        "cpu": [
            "faiss-cpu>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "reid-extract=inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json"],
    },
)
