#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
카메라 간 사람 매칭 (Pure Matching Module)
- 외부에서 임베딩을 추출하여 본 모듈에 전달하면, 유사도 계산과 매칭/통합ID 매핑만 수행합니다.
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import logging


class CrossCameraMatcher:
    """임베딩 기반 카메라 간 매칭 전용 클래스
        
        Args:
        threshold: 임계값 (cosine: 최소 유사도, euclidean: 최대 거리)
        metric: 'cosine' 또는 'euclidean'
    """

    def __init__(self, threshold: float = 0.7, metric: str = 'cosine'):
        self.threshold = threshold
        self.metric = metric.lower()
        self.logger = self._setup_logger()
        self.matches: Dict[int, List[Tuple[str, int]]] = {}
        self.unmatched_tracks: List[Tuple[str, int]] = []
        self.unified_ids: Dict[str, Dict[int, int]] = {}
    
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger('cross_camera_matcher')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        return logger
    
    def compute_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        유사도/거리 행렬 계산 (metric에 따라 다름)

        - cosine: 코사인 유사도 행렬 (값↑ 좋음)
        - euclidean: 유클리디안 거리 행렬 (값↓ 좋음)
        """
        if embeddings.ndim != 2:
            raise ValueError('embeddings must be a 2D array of shape (N, D)')

        if self.metric == 'cosine':
            eps = 1e-12
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + eps
            feats = embeddings / norms
            return feats @ feats.T

        if self.metric == 'euclidean':
            x = embeddings
            x2 = (x ** 2).sum(axis=1, keepdims=True)
            dist2 = x2 + x2.T - 2.0 * (x @ x.T)
            dist2 = np.clip(dist2, a_min=0.0, a_max=None)
            return np.sqrt(dist2 + 1e-12)

        raise ValueError("metric must be one of ['cosine', 'euclidean']")

    def find_matches(self, matrix: np.ndarray, all_ids: List[Tuple[str, int]]) -> Dict[int, List[Tuple[str, int]]]:
        """
        유사도/거리 행렬을 기반으로 매칭 그룹 생성 (metric에 따라 비교 방식 상이)
        
        Args:
            matrix: (N, N) 유사도/거리 행렬
            all_ids: 길이 N의 (floor, track_id) 목록
            
        Returns:
            {group_id: [(floor, track_id), ...]} 딕셔너리
        """
        self.logger.info("매칭 찾기 시작...")
        
        n = len(all_ids)
        matches = defaultdict(list)
        used = set()
        
        for i in range(n):
            if i in used:
                continue
            floor_i, track_i = all_ids[i]
            current_group = [(floor_i, track_i)]
            
            for j in range(i + 1, n):
                if j in used:
                    continue
                if (self.metric == 'cosine' and matrix[i, j] >= self.threshold) or \
                   (self.metric == 'euclidean' and matrix[i, j] <= self.threshold):
                    floor_j, track_j = all_ids[j]
                    current_group.append((floor_j, track_j))
                    used.add(j)
            
            if len(current_group) > 1:
                matches[len(matches)] = current_group
                used.add(i)
        
        # 매칭되지 않은 단일 track도 통합 ID 부여를 위해 그룹으로 포함
        unmatched_tracks: List[Tuple[str, int]] = []
        for i in range(n):
            if i not in used:
                floor_i, track_i = all_ids[i]
                unmatched_tracks.append((floor_i, track_i))
                matches[len(matches)] = [(floor_i, track_i)]
        
        self.logger.info(f"총 {len(matches)}개 그룹 발견")
        self.logger.info(f"매칭되지 않은 track: {len(unmatched_tracks)}개")
        
        self.unmatched_tracks = unmatched_tracks
        self.matches = dict(matches)
        return self.matches
    
    def create_unified_mapping(self) -> Dict[str, Dict[int, int]]:
        """
        매칭 결과(self.matches)를 통합 ID 매핑으로 변환
        
        Returns:
            {floor: {track_id: unified_id}}
        """
        self.logger.info("통합 ID 매핑 생성...")
        unified_ids: Dict[str, Dict[int, int]] = {}

        unified_id_counter = 1
        for _, tracks in self.matches.items():
            unified_id = unified_id_counter
            unified_id_counter += 1
            for floor, track_id in tracks:
                unified_ids.setdefault(floor, {})[track_id] = unified_id
        
        self.unified_ids = unified_ids
        self.logger.info(f"통합 ID 매핑 생성 완료: {unified_id_counter - 1}개 고유 ID")
        return unified_ids
        
    def match_from_embeddings_dict(self, embeddings_by_floor: Dict[str, Dict[int, np.ndarray]]):
        """
        편의 함수: 층/트랙별 임베딩 딕셔너리를 받아 전체 매칭 수행
        
        Args:
            embeddings_by_floor: {floor: {track_id: embedding(np.ndarray)}}

        Returns:
            similarity_matrix, all_ids, matches, unified_ids
        """
        all_embeddings: List[np.ndarray] = []
        all_ids: List[Tuple[str, int]] = []

        for floor, track_dict in embeddings_by_floor.items():
            for track_id, emb in track_dict.items():
                all_embeddings.append(np.asarray(emb))
                all_ids.append((floor, track_id))

        if not all_embeddings:
            raise ValueError('빈 임베딩 입력입니다.')

        embeddings_array = np.stack(all_embeddings, axis=0)
        mat = self.compute_matrix(embeddings_array)
        matches = self.find_matches(mat, all_ids)
        unified = self.create_unified_mapping()
        return mat, all_ids, matches, unified