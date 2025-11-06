#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReID Embedding Extractor Package
SOLIDER_REID 모델을 사용한 ReID 임베딩 추출 도구
"""

# SOLIDER 모델에서 주요 클래스 import
from .models import SOLIDEREmbeddingExtractor

__version__ = "1.0.0"
__author__ = "ReID Team"
__email__ = "reid@example.com"

__all__ = [
    "SOLIDEREmbeddingExtractor",
]
