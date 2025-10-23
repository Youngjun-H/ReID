#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReID Embedding Extractor Package
SOLIDER_REID 모델을 사용한 ReID 임베딩 추출 도구
"""

from .embedding_extractor import ReIDEmbeddingExtractor
from .config import create_default_config

__version__ = "1.0.0"
__author__ = "ReID Team"
__email__ = "reid@example.com"

__all__ = [
    "ReIDEmbeddingExtractor",
    "create_default_config",
]
