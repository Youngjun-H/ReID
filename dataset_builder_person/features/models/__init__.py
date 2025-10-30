# Models package for independent ReID embedding extractor
from .solider import SOLIDEREmbeddingExtractor
from .osnet import OSNetFeatureExtractor, compute_pairwise_similarity, compare_images

__all__ = [
    'SOLIDEREmbeddingExtractor',
    'OSNetFeatureExtractor',
    'compute_pairwise_similarity',
    'compare_images',
]
