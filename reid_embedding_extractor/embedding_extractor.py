#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ReID Embedding Extractor - 독립적인 버전
SOLIDER_REID 모델을 사용하여 사람 이미지에서 ReID 임베딩을 추출하는 최적화된 클래스
SOLIDER_REID 의존성을 제거하고 완전히 독립적으로 작동
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import numpy as np
from typing import Union, List, Tuple, Optional
from pathlib import Path
import logging

# 독립적인 모듈들 import
try:
    # 패키지로 import될 때 (상대 import)
    from .config import cfg
    from .model_factory import make_model
    from .utils import setup_logger
except ImportError:
    # 직접 실행될 때 (절대 import)
    from config import cfg
    from model_factory import make_model
    from utils import setup_logger


class ReIDEmbeddingExtractor:
    """
    SOLIDER_REID 모델을 사용하여 사람 이미지에서 ReID 임베딩을 추출하는 최적화된 클래스
    PyTorch 최신 버전 호환
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: Optional[str] = None,
                 device: Optional[str] = None,
                 semantic_weight: float = 0.2,
                 image_size: Tuple[int, int] = (384, 128),
                 normalize_features: bool = True):
        """
        ReID 임베딩 추출기 초기화
        
        Args:
            model_path (str): 훈련된 SOLIDER_REID 모델 경로
            config_path (Optional[str]): 설정 파일 경로 (None이면 기본 설정 사용)
            device (Optional[str]): 사용할 디바이스 (None이면 자동 선택)
            semantic_weight (float): 시맨틱 가중치 (기본값: 0.2)
            image_size (Tuple[int, int]): 입력 이미지 크기 (height, width)
            normalize_features (bool): 특징 벡터 L2 정규화 여부
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.semantic_weight = semantic_weight
        self.image_size = image_size
        self.normalize_features = normalize_features
        
        # 로거 설정
        self.logger = logging.getLogger("reid_embedding")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # 설정 로드
        self._load_config(config_path)
        
        # 모델 로드
        self._load_model(model_path)
        
        # 전처리 변환 설정
        self._setup_transforms()
        
        self.logger.info(f"ReID Embedding Extractor 초기화 완료")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Image size: {self.image_size}")
        self.logger.info(f"Semantic weight: {self.semantic_weight}")
        self.logger.info(f"Normalize features: {self.normalize_features}")
    
    def _load_config(self, config_path: Optional[str]):
        """설정 파일 로드"""
        if config_path and os.path.exists(config_path):
            cfg.merge_from_file(config_path)
            self.logger.info(f"설정 파일 로드: {config_path}")
        else:
            # 모델 이름으로 설정 파일 자동 찾기
            try:
                from .config import find_config_file, get_available_configs
            except ImportError:
                from config import find_config_file, get_available_configs
            
            # 모델 경로에서 모델 이름 추출 시도
            model_name = None
            if hasattr(self, 'model_path') and self.model_path:
                model_path = Path(self.model_path)
                # 파일명에서 모델 타입 추출 (예: swin_base_model.pth -> swin_base)
                for name in ['swin_base', 'swin_tiny', 'swin_small']:
                    if name in model_path.stem:
                        model_name = name
                        break
            
            if model_name:
                auto_config_path = find_config_file(model_name)
                if auto_config_path:
                    cfg.merge_from_file(auto_config_path)
                    self.logger.info(f"자동 설정 파일 로드: {auto_config_path}")
                else:
                    self.logger.info(f"모델 '{model_name}'에 대한 설정 파일을 찾을 수 없습니다. 기본 설정 사용")
                    available_configs = get_available_configs()
                    if available_configs:
                        self.logger.info(f"사용 가능한 설정: {', '.join(available_configs)}")
            else:
                # 기본 설정 사용
                self.logger.info("기본 설정 사용")
        
        # 필수 설정 적용
        cfg.MODEL.semantic_weight = self.semantic_weight
        cfg.INPUT.size_train = list(self.image_size)
        cfg.INPUT.size_test = list(self.image_size)
        cfg.freeze()
    
    def _load_model(self, model_path: str):
        """모델 로드"""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
        
        # 더미 데이터로 모델 생성 (실제 데이터셋 정보는 필요하지 않음)
        num_classes = 1000  # 더미 클래스 수
        camera_num = 6      # 더미 카메라 수
        view_num = 1        # 더미 뷰 수
        
        # 모델 생성
        self.model = make_model(
            cfg, 
            num_class=num_classes, 
            camera_num=camera_num, 
            view_num=view_num, 
            semantic_weight=self.semantic_weight
        )
        
        # 모델 가중치 로드
        self.model.load_param(str(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        self.logger.info(f"모델 로드 완료: {model_path}")
    
    def _setup_transforms(self):
        """이미지 전처리 변환 설정"""
        self.transform = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.pixel_mean, std=cfg.INPUT.pixel_std)
        ])
        
        self.logger.info("이미지 전처리 변환 설정 완료")
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        이미지 전처리
        
        Args:
            image: 이미지 경로, PIL Image, 또는 numpy array
            
        Returns:
            torch.Tensor: 전처리된 이미지 텐서
        """
        if isinstance(image, str):
            # 파일 경로인 경우
            if not os.path.exists(image):
                raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image}")
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # numpy array인 경우
            image = Image.fromarray(image).convert('RGB')
        elif not isinstance(image, Image.Image):
            raise ValueError("지원되지 않는 이미지 타입입니다.")
        
        # 전처리 적용
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # 배치 차원 추가
        
        return image_tensor
    
    @torch.inference_mode()
    def extract_embedding(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        단일 이미지에서 임베딩 추출
        
        Args:
            image: 이미지 경로, PIL Image, 또는 numpy array
            
        Returns:
            np.ndarray: 추출된 임베딩 벡터
        """
        # 이미지 전처리
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # 임베딩 추출
        if hasattr(self.model, 'base') and hasattr(self.model.base, 'forward'):
            # Transformer 모델인 경우
            global_feat, _ = self.model(image_tensor)
        else:
            # ResNet 모델인 경우
            global_feat = self.model(image_tensor)
        
        # CPU로 이동하고 numpy로 변환
        embedding = global_feat.cpu().numpy().flatten()
        
        # L2 정규화 (선택사항)
        if self.normalize_features:
            embedding = self._normalize_feature(embedding)
        
        return embedding
    
    @torch.inference_mode()
    def extract_embeddings_batch(self, images: List[Union[str, Image.Image, np.ndarray]], 
                                batch_size: int = 32) -> List[np.ndarray]:
        """
        여러 이미지에서 배치로 임베딩 추출
        
        Args:
            images: 이미지 리스트
            batch_size: 배치 크기
            
        Returns:
            List[np.ndarray]: 임베딩 리스트
        """
        embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_tensors = []
            valid_indices = []
            
            # 배치 이미지 전처리
            for j, image in enumerate(batch_images):
                try:
                    image_tensor = self.preprocess_image(image)
                    batch_tensors.append(image_tensor)
                    valid_indices.append(j)
                except Exception as e:
                    self.logger.warning(f"이미지 처리 실패 (인덱스 {j}): {e}")
                    # 더미 임베딩으로 대체
                    dummy_embedding = np.zeros(2048, dtype=np.float32)
                    embeddings.append(dummy_embedding)
                    continue
            
            if not batch_tensors:
                continue
            
            # 배치 텐서 결합
            batch_tensor = torch.cat(batch_tensors, dim=0)
            batch_tensor = batch_tensor.to(self.device)
            
            # 배치 임베딩 추출
            if hasattr(self.model, 'base') and hasattr(self.model.base, 'forward'):
                global_feat, _ = self.model(batch_tensor)
            else:
                global_feat = self.model(batch_tensor)
            
            # CPU로 이동하고 numpy로 변환
            batch_embeddings = global_feat.cpu().numpy()
            
            # L2 정규화 (선택사항)
            if self.normalize_features:
                batch_embeddings = self._normalize_features_batch(batch_embeddings)
            
            # 유효한 임베딩만 추가
            for k, idx in enumerate(valid_indices):
                embeddings.append(batch_embeddings[k].flatten())
        
        return embeddings
    
    def _normalize_feature(self, feature: np.ndarray) -> np.ndarray:
        """단일 특징 벡터 L2 정규화"""
        norm = np.linalg.norm(feature)
        if norm < 1e-12:
            return feature
        return feature / norm
    
    def _normalize_features_batch(self, features: np.ndarray) -> np.ndarray:
        """배치 특징 벡터 L2 정규화"""
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        return features / norms
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        두 임베딩 간의 유사도 계산 (코사인 유사도)
        
        Args:
            embedding1: 첫 번째 임베딩
            embedding2: 두 번째 임베딩
            
        Returns:
            float: 코사인 유사도
        """
        # 코사인 유사도 계산
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 < 1e-12 or norm2 < 1e-12:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def find_most_similar(self, query_embedding: np.ndarray, 
                         gallery_embeddings: List[np.ndarray]) -> Tuple[int, float]:
        """
        갤러리에서 가장 유사한 임베딩 찾기
        
        Args:
            query_embedding: 쿼리 임베딩
            gallery_embeddings: 갤러리 임베딩 리스트
            
        Returns:
            Tuple[int, float]: 가장 유사한 인덱스와 유사도
        """
        similarities = []
        for gallery_emb in gallery_embeddings:
            sim = self.compute_similarity(query_embedding, gallery_emb)
            similarities.append(sim)
        
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        return best_idx, best_similarity
    
    def compute_similarity_matrix(self, embeddings1: List[np.ndarray], 
                                 embeddings2: List[np.ndarray]) -> np.ndarray:
        """
        두 임베딩 세트 간의 유사도 행렬 계산
        
        Args:
            embeddings1: 첫 번째 임베딩 세트
            embeddings2: 두 번째 임베딩 세트
            
        Returns:
            np.ndarray: 유사도 행렬 (len(embeddings1) x len(embeddings2))
        """
        embeddings1 = np.array(embeddings1)
        embeddings2 = np.array(embeddings2)
        
        # 정규화
        if self.normalize_features:
            embeddings1 = self._normalize_features_batch(embeddings1)
            embeddings2 = self._normalize_features_batch(embeddings2)
        
        # 코사인 유사도 행렬 계산
        similarity_matrix = np.dot(embeddings1, embeddings2.T)
        
        return similarity_matrix


def create_default_config() -> str:
    """기본 설정 파일 내용 생성"""
    config_content = """MODEL:
  PRETRAIN_HW_RATIO: 2
  METRIC_LOSS_TYPE: 'triplet'
  IF_LABELSMOOTH: 'off'
  IF_WITH_CENTER: 'no'
  NAME: 'transformer'
  NO_MARGIN: True
  DEVICE_ID: ('0')
  TRANSFORMER_TYPE: 'swin_base_patch4_window7_224'
  STRIDE_SIZE: [16, 16]

INPUT:
  SIZE_TRAIN: [384, 128]
  SIZE_TEST: [384, 128]
  PROB: 0.5
  RE_PROB: 0.5
  PADDING: 10
  PIXEL_MEAN: [0.5, 0.5, 0.5]
  PIXEL_STD: [0.5, 0.5, 0.5]

DATASETS:
  NAMES: ('market1501')
  ROOT_DIR: ('../data')

DATALOADER:
  SAMPLER: 'softmax_triplet'
  NUM_INSTANCE: 4
  NUM_WORKERS: 8

SOLVER:
  OPTIMIZER_NAME: 'SGD'
  MAX_EPOCHS: 120
  BASE_LR: 0.0008
  WARMUP_EPOCHS: 20
  IMS_PER_BATCH: 64
  WARMUP_METHOD: 'cosine'
  LARGE_FC_LR: False
  CHECKPOINT_PERIOD: 120
  LOG_PERIOD: 20
  EVAL_PERIOD: 10
  WEIGHT_DECAY: 1e-4
  WEIGHT_DECAY_BIAS: 1e-4
  BIAS_LR_FACTOR: 2

TEST:
  EVAL: True
  IMS_PER_BATCH: 256
  RE_RANKING: False
  WEIGHT: ''
  NECK_FEAT: 'before'
  FEAT_NORM: 'yes'

OUTPUT_DIR: './log/default'
"""
    return config_content


if __name__ == "__main__":
    # 사용 예제
    print("ReID Embedding Extractor - PyTorch 최신 버전 호환")
    print("=" * 60)
    
    # 기본 설정 파일 생성
    config_content = create_default_config()
    with open("default_config.yml", "w") as f:
        f.write(config_content)
    print("기본 설정 파일 생성: default_config.yml")