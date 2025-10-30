#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
카메라 간 사람 매칭 시스템
층별로 추적된 사람 ID들을 ReID 임베딩을 사용하여 통합 ID로 매핑
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import logging
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import argparse
from datetime import datetime
import shutil

# ReID 임베딩 추출기 임포트
import sys
sys.path.append('/data/reid/reid_master/reid_embedding_extractor')
from models import SOLIDEREmbeddingExtractor


class CrossCameraMatcher:
    """카메라 간 사람 매칭 클래스"""
    
    def __init__(self, 
                 tracklets_dir: str,
                 model_path: str,
                 config_path: Optional[str] = None,
                 similarity_threshold: float = 0.7,
                 device: str = 'cuda'):
        """
        초기화
        
        Args:
            tracklets_dir: tracklets 폴더 경로
            model_path: ReID 모델 경로
            config_path: 모델 설정 파일 경로
            similarity_threshold: 유사도 임계값
            device: 사용할 디바이스
        """
        self.tracklets_dir = Path(tracklets_dir)
        self.similarity_threshold = similarity_threshold
        self.device = device
        
        # 로거 설정
        self.logger = self._setup_logger()
        
        # ReID 임베딩 추출기 초기화
        self.logger.info("ReID 임베딩 추출기 초기화...")
        self.extractor = SOLIDEREmbeddingExtractor(
            model_path=model_path,
            config_path=config_path,
            device=device,
            semantic_weight=0.2,
            image_size=(384, 128),
            normalize_features=True
        )
        
        # 데이터 저장소
        self.track_data = {}  # {floor: {track_id: track_info}}
        self.embeddings = {}  # {floor: {track_id: embedding}}
        self.matches = {}     # 매칭 결과
        self.unified_ids = {} # 통합된 ID 매핑
        
        self.logger.info("CrossCameraMatcher 초기화 완료")
    
    def _setup_logger(self) -> logging.Logger:
        """로거 설정"""
        logger = logging.getLogger('cross_camera_matcher')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # 콘솔 핸들러
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            
            # 포맷터
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def load_tracklets(self) -> Dict[str, Dict]:
        """
        모든 층의 tracklets 데이터 로드
        
        Returns:
            Dict[str, Dict]: {floor: {track_id: track_info}}
        """
        self.logger.info("Tracklets 데이터 로드 시작...")
        
        floors = ['1F', '2F', '3F']
        
        for floor in floors:
            floor_dir = self.tracklets_dir / floor
            if not floor_dir.exists():
                self.logger.warning(f"층 {floor} 디렉토리가 존재하지 않습니다: {floor_dir}")
                continue
            
            self.track_data[floor] = {}
            
            # 각 track 폴더 처리
            for track_dir in sorted(floor_dir.iterdir()):
                if not track_dir.is_dir() or not track_dir.name.startswith('track_'):
                    continue
                
                track_id = int(track_dir.name.split('_')[1])
                metadata_path = track_dir / 'metadata.json'
                
                if not metadata_path.exists():
                    self.logger.warning(f"메타데이터 파일이 없습니다: {metadata_path}")
                    continue
                
                # 메타데이터 로드
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 이미지 파일들 확인
                image_files = []
                for img_path in metadata.get('image_paths', []):
                    # 상대 경로를 절대 경로로 변환
                    if img_path.startswith('tracklets/'):
                        # tracklets/로 시작하는 경우 절대 경로로 변환
                        full_path = self.tracklets_dir.parent / img_path
                    else:
                        # 이미 절대 경로인 경우
                        full_path = Path(img_path)
                    
                    if full_path.exists():
                        image_files.append(full_path)
                
                if not image_files:
                    self.logger.warning(f"이미지 파일이 없습니다: {track_dir}")
                    continue
                
                # track 정보 저장
                self.track_data[floor][track_id] = {
                    'metadata': metadata,
                    'image_files': image_files,
                    'track_dir': track_dir
                }
            
            self.logger.info(f"층 {floor}: {len(self.track_data[floor])}개 track 로드 완료")
        
        total_tracks = sum(len(tracks) for tracks in self.track_data.values())
        self.logger.info(f"총 {total_tracks}개 track 로드 완료")
        
        return self.track_data
    
    def extract_embeddings(self) -> Dict[str, Dict]:
        """
        모든 track의 임베딩 추출
        
        Returns:
            Dict[str, Dict]: {floor: {track_id: embedding}}
        """
        self.logger.info("임베딩 추출 시작...")
        
        for floor, tracks in self.track_data.items():
            self.logger.info(f"층 {floor} 임베딩 추출 중...")
            self.embeddings[floor] = {}
            
            for track_id, track_info in tracks.items():
                try:
                    # 대표 이미지 선택 (중간 프레임)
                    image_files = track_info['image_files']
                    if not image_files:
                        continue
                    
                    # 중간 프레임 선택
                    middle_idx = len(image_files) // 2
                    representative_image = image_files[middle_idx]
                    
                    # 임베딩 추출
                    embedding = self.extractor.extract_embedding(str(representative_image))
                    self.embeddings[floor][track_id] = embedding
                    
                    self.logger.debug(f"층 {floor}, Track {track_id}: 임베딩 추출 완료")
                    
                except Exception as e:
                    self.logger.error(f"층 {floor}, Track {track_id} 임베딩 추출 실패: {e}")
                    continue
            
            self.logger.info(f"층 {floor}: {len(self.embeddings[floor])}개 임베딩 추출 완료")
        
        total_embeddings = sum(len(embeddings) for embeddings in self.embeddings.values())
        self.logger.info(f"총 {total_embeddings}개 임베딩 추출 완료")
        
        return self.embeddings
    
    def compute_similarity_matrix(self) -> np.ndarray:
        """
        모든 층 간 유사도 행렬 계산
        
        Returns:
            np.ndarray: 유사도 행렬
        """
        self.logger.info("유사도 행렬 계산 시작...")
        
        # 모든 임베딩과 ID 정보 수집
        all_embeddings = []
        all_ids = []  # (floor, track_id) 튜플
        
        for floor, embeddings in self.embeddings.items():
            for track_id, embedding in embeddings.items():
                all_embeddings.append(embedding)
                all_ids.append((floor, track_id))
        
        if not all_embeddings:
            self.logger.error("임베딩이 없습니다.")
            return np.array([])
        
        # 유사도 행렬 계산
        embeddings_array = np.array(all_embeddings)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        self.logger.info(f"유사도 행렬 계산 완료: {similarity_matrix.shape}")
        
        return similarity_matrix, all_ids
    
    def find_matches(self, similarity_matrix: np.ndarray, all_ids: List[Tuple[str, int]]) -> Dict:
        """
        유사도 행렬을 기반으로 매칭 찾기
        
        Args:
            similarity_matrix: 유사도 행렬
            all_ids: ID 리스트
            
        Returns:
            Dict: 매칭 결과
        """
        self.logger.info("매칭 찾기 시작...")
        
        n = len(all_ids)
        matches = defaultdict(list)
        used = set()
        
        # 유사도가 임계값 이상인 쌍 찾기
        for i in range(n):
            if i in used:
                continue
                
            floor_i, track_i = all_ids[i]
            current_group = [(floor_i, track_i)]
            
            for j in range(i + 1, n):
                if j in used:
                    continue
                    
                if similarity_matrix[i, j] >= self.similarity_threshold:
                    floor_j, track_j = all_ids[j]
                    current_group.append((floor_j, track_j))
                    used.add(j)
            
            if len(current_group) > 1:  # 2개 이상의 track이 매칭된 경우만
                matches[len(matches)] = current_group
                used.add(i)  # 매칭된 그룹에 속한 track만 used에 추가
        
        # 매칭되지 않은 단일 track들 (별도 처리)
        unmatched_tracks = []
        for i in range(n):
            if i not in used:
                floor_i, track_i = all_ids[i]
                unmatched_tracks.append((floor_i, track_i))
                # 단일 track도 matches에 추가 (통합 ID 할당을 위해)
                matches[len(matches)] = [(floor_i, track_i)]
        
        self.logger.info(f"총 {len(matches)}개 그룹 발견")
        self.logger.info(f"매칭되지 않은 track: {len(unmatched_tracks)}개")
        
        # 매칭되지 않은 track들 저장
        self.unmatched_tracks = unmatched_tracks
        
        # 매칭 결과 저장
        self.matches = dict(matches)
        
        return self.matches
    
    def create_unified_mapping(self) -> Dict[str, Dict[int, int]]:
        """
        통합 ID 매핑 생성
        
        Returns:
            Dict[str, Dict[int, int]]: {floor: {original_track_id: unified_id}}
        """
        self.logger.info("통합 ID 매핑 생성...")
        
        self.unified_ids = {}
        
        # 각 층별 매핑 초기화
        for floor in self.track_data.keys():
            self.unified_ids[floor] = {}
        
        # 매칭 결과를 기반으로 통합 ID 할당
        unified_id_counter = 1
        
        for group_id, tracks in self.matches.items():
            unified_id = unified_id_counter
            unified_id_counter += 1
            
            for floor, track_id in tracks:
                self.unified_ids[floor][track_id] = unified_id
        
        self.logger.info(f"통합 ID 매핑 생성 완료: {unified_id_counter - 1}개 고유 ID")
        
        return self.unified_ids
    
    def save_results(self, output_dir: str):
        """
        결과 저장
        
        Args:
            output_dir: 출력 디렉토리
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 통합 ID 매핑 저장
        mapping_file = output_path / 'unified_id_mapping.json'
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump(self.unified_ids, f, indent=2, ensure_ascii=False)
        
        # 매칭 결과 저장
        matches_file = output_path / 'matches.json'
        with open(matches_file, 'w', encoding='utf-8') as f:
            json.dump(self.matches, f, indent=2, ensure_ascii=False)
        
        # 통계 정보 저장
        stats = self._generate_statistics()
        stats_file = output_path / 'statistics.json'
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"결과 저장 완료: {output_path}")
    
    def _generate_statistics(self) -> Dict:
        """통계 정보 생성"""
        stats = {
            'total_tracks': sum(len(tracks) for tracks in self.track_data.values()),
            'total_unified_ids': len(self.matches),
            'floors': list(self.track_data.keys()),
            'tracks_per_floor': {floor: len(tracks) for floor, tracks in self.track_data.items()},
            'matches_per_floor': {},
            'unified_ids_per_floor': {}
        }
        
        # 층별 매칭 통계
        for floor in self.track_data.keys():
            floor_matches = 0
            floor_unified_ids = set()
            
            for group_id, tracks in self.matches.items():
                floor_tracks = [t for t in tracks if t[0] == floor]
                if floor_tracks:
                    floor_matches += len(floor_tracks)
                    floor_unified_ids.add(group_id)
            
            stats['matches_per_floor'][floor] = floor_matches
            stats['unified_ids_per_floor'][floor] = len(floor_unified_ids)
        
        return stats
    
    def visualize_matches(self, output_dir: str, max_groups: int = 10):
        """
        매칭 결과 시각화
        
        Args:
            output_dir: 출력 디렉토리
            max_groups: 시각화할 최대 그룹 수
        """
        self.logger.info("매칭 결과 시각화...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 매칭된 그룹들만 시각화 (단일 track 제외)
        multi_track_groups = {k: v for k, v in self.matches.items() if len(v) > 1}
        
        if not multi_track_groups:
            self.logger.warning("매칭된 그룹이 없습니다.")
            return
        
        # 시각화할 그룹 수 제한
        groups_to_visualize = list(multi_track_groups.items())[:max_groups]
        
        for group_id, tracks in groups_to_visualize:
            self._visualize_group(group_id, tracks, output_path)
        
        self.logger.info(f"시각화 완료: {output_path}")
    
    def _visualize_group(self, group_id: int, tracks: List[Tuple[str, int]], output_path: Path):
        """개별 그룹 시각화"""
        n_tracks = len(tracks)
        fig, axes = plt.subplots(1, n_tracks, figsize=(4 * n_tracks, 4))
        
        if n_tracks == 1:
            axes = [axes]
        
        for i, (floor, track_id) in enumerate(tracks):
            track_info = self.track_data[floor][track_id]
            image_files = track_info['image_files']
            
            # 대표 이미지 선택
            middle_idx = len(image_files) // 2
            representative_image = image_files[middle_idx]
            
            # 이미지 로드 및 표시
            img = Image.open(representative_image)
            axes[i].imshow(img)
            axes[i].set_title(f"{floor} Track {track_id}")
            axes[i].axis('off')
        
        plt.suptitle(f"Group {group_id} - {n_tracks} tracks")
        plt.tight_layout()
        
        # 저장
        output_file = output_path / f"group_{group_id:03d}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def copy_images_by_unified_id(self, output_dir: str):
        """
        통합 ID별로 이미지들을 복사하여 폴더 구조 생성
        
        Args:
            output_dir: 출력 디렉토리
        """
        self.logger.info("통합 ID별 이미지 복사 시작...")
        
        output_path = Path(output_dir)
        unified_images_dir = output_path / 'unified_images'
        unified_images_dir.mkdir(parents=True, exist_ok=True)
        
        # 각 통합 ID별로 폴더 생성 및 이미지 복사
        for group_id, tracks in self.matches.items():
            unified_id = group_id + 1  # 1부터 시작하는 ID
            group_dir = unified_images_dir / f"unified_id_{unified_id:03d}"
            group_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"통합 ID {unified_id}: {len(tracks)}개 track 처리 중...")
            
            # 각 track의 이미지들을 복사
            for floor, track_id in tracks:
                track_info = self.track_data[floor][track_id]
                image_files = track_info['image_files']
                
                # 각 이미지를 복사
                for i, image_path in enumerate(image_files):
                    # 파일명 생성: floor_trackid_frameindex.jpg
                    original_name = image_path.name
                    new_name = f"{floor}_{track_id:03d}_{i:03d}_{original_name}"
                    dest_path = group_dir / new_name
                    
                    try:
                        shutil.copy2(image_path, dest_path)
                    except Exception as e:
                        self.logger.warning(f"이미지 복사 실패: {image_path} -> {dest_path}, 오류: {e}")
            
            # 그룹 정보 파일 생성
            group_info = {
                'unified_id': unified_id,
                'group_id': group_id,
                'tracks': tracks,
                'total_images': sum(len(self.track_data[track[0]][track[1]]['image_files']) 
                                  for track in tracks),
                'floors': list(set(track[0] for track in tracks))
            }
            
            info_file = group_dir / 'group_info.json'
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(group_info, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"통합 ID별 이미지 복사 완료: {unified_images_dir}")
        
        # 전체 통계 파일 생성
        self._create_unified_images_summary(unified_images_dir)
        
        return unified_images_dir
    
    def _create_unified_images_summary(self, unified_images_dir: Path):
        """통합 이미지 폴더 요약 정보 생성"""
        summary = {
            'total_unified_ids': len(self.matches),
            'total_tracks': sum(len(tracks) for tracks in self.matches.values()),
            'total_images': 0,
            'unified_ids': []
        }
        
        for group_id, tracks in self.matches.items():
            unified_id = group_id + 1
            group_dir = unified_images_dir / f"unified_id_{unified_id:03d}"
            
            # 이미지 개수 계산
            image_count = 0
            for floor, track_id in tracks:
                track_info = self.track_data[floor][track_id]
                image_count += len(track_info['image_files'])
            
            summary['total_images'] += image_count
            summary['unified_ids'].append({
                'unified_id': unified_id,
                'group_id': group_id,
                'tracks': tracks,
                'image_count': image_count,
                'floors': list(set(track[0] for track in tracks))
            })
        
        # 요약 파일 저장
        summary_file = unified_images_dir / 'summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"통합 이미지 요약 생성: {summary_file}")
    
    def copy_unmatched_images(self, output_dir: str):
        """
        매칭되지 않은 track들의 이미지를 별도 폴더에 저장
        
        Args:
            output_dir: 출력 디렉토리
        """
        if not hasattr(self, 'unmatched_tracks') or not self.unmatched_tracks:
            self.logger.info("매칭되지 않은 track이 없습니다.")
            return
        
        self.logger.info(f"매칭되지 않은 track 이미지 복사 시작... ({len(self.unmatched_tracks)}개)")
        
        output_path = Path(output_dir)
        unmatched_images_dir = output_path / 'unmatched_images'
        unmatched_images_dir.mkdir(parents=True, exist_ok=True)
        
        # 매칭되지 않은 track들을 개별 폴더로 저장
        for idx, (floor, track_id) in enumerate(self.unmatched_tracks):
            track_info = self.track_data[floor][track_id]
            image_files = track_info['image_files']
            
            # 개별 track 폴더 생성
            track_dir = unmatched_images_dir / f"{floor}_track_{track_id:03d}"
            track_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"매칭되지 않은 track {floor}_{track_id:03d}: {len(image_files)}개 이미지 처리 중...")
            
            # 각 이미지를 복사
            for i, image_path in enumerate(image_files):
                # 파일명 생성: frame_index_원본파일명
                original_name = image_path.name
                new_name = f"{i:03d}_{original_name}"
                dest_path = track_dir / new_name
                
                try:
                    shutil.copy2(image_path, dest_path)
                except Exception as e:
                    self.logger.warning(f"이미지 복사 실패: {image_path} -> {dest_path}, 오류: {e}")
            
            # track 정보 파일 생성
            track_info_data = {
                'floor': floor,
                'track_id': track_id,
                'total_images': len(image_files),
                'image_files': [str(img) for img in image_files],
                'metadata': track_info['metadata']
            }
            
            info_file = track_dir / 'track_info.json'
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(track_info_data, f, indent=2, ensure_ascii=False)
        
        # 매칭되지 않은 track들 요약 생성
        self._create_unmatched_summary(unmatched_images_dir)
        
        self.logger.info(f"매칭되지 않은 track 이미지 복사 완료: {unmatched_images_dir}")
        
        return unmatched_images_dir
    
    def _create_unmatched_summary(self, unmatched_images_dir: Path):
        """매칭되지 않은 track들 요약 정보 생성"""
        summary = {
            'total_unmatched_tracks': len(self.unmatched_tracks),
            'total_images': 0,
            'unmatched_tracks': []
        }
        
        for floor, track_id in self.unmatched_tracks:
            track_info = self.track_data[floor][track_id]
            image_count = len(track_info['image_files'])
            
            summary['total_images'] += image_count
            summary['unmatched_tracks'].append({
                'floor': floor,
                'track_id': track_id,
                'image_count': image_count,
                'folder_name': f"{floor}_track_{track_id:03d}"
            })
        
        # 요약 파일 저장
        summary_file = unmatched_images_dir / 'unmatched_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"매칭되지 않은 track 요약 생성: {summary_file}")
    
    def visualize_similarity_matrix(self, similarity_matrix: np.ndarray, all_ids: List[Tuple[str, int]], output_dir: str):
        """
        유사도 행렬을 시각화하여 이미지로 저장
        
        Args:
            similarity_matrix: 유사도 행렬
            all_ids: ID 리스트 [(floor, track_id), ...]
            output_dir: 출력 디렉토리
        """
        self.logger.info("유사도 행렬 시각화 시작...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 전체 유사도 행렬 히트맵
        plt.figure(figsize=(15, 12))
        
        # 히트맵 생성
        im = plt.imshow(similarity_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        
        # 축 레이블 생성
        labels = [f"{floor}_{track_id:03d}" for floor, track_id in all_ids]
        
        # 축 설정
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(range(len(labels)), labels)
        plt.xlabel('Track ID')
        plt.ylabel('Track ID')
        plt.title('Similarity Matrix Heatmap')
        
        # 컬러바 추가
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
        
        # 격자 추가
        plt.grid(True, alpha=0.3)
        
        # 레이아웃 조정
        plt.tight_layout()
        
        # 저장
        heatmap_file = output_path / 'similarity_matrix_heatmap.png'
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"유사도 행렬 히트맵 저장: {heatmap_file}")
        
        # 2. 층별 유사도 행렬들
        floors = ['1F', '2F', '3F']
        floor_indices = {floor: [] for floor in floors}
        
        # 각 층별 인덱스 수집
        for i, (floor, track_id) in enumerate(all_ids):
            floor_indices[floor].append(i)
        
        # 층별 유사도 행렬 생성 및 시각화
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, floor in enumerate(floors):
            if not floor_indices[floor]:
                axes[idx].text(0.5, 0.5, f'No tracks in {floor}', 
                             ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{floor} Similarity Matrix')
                continue
            
            # 해당 층의 인덱스들
            indices = floor_indices[floor]
            
            # 층별 유사도 행렬 추출
            floor_matrix = similarity_matrix[np.ix_(indices, indices)]
            
            # 히트맵 생성
            im = axes[idx].imshow(floor_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
            
            # 축 레이블
            floor_labels = [f"{track_id:03d}" for _, track_id in [all_ids[i] for i in indices]]
            axes[idx].set_xticks(range(len(floor_labels)))
            axes[idx].set_yticks(range(len(floor_labels)))
            axes[idx].set_xticklabels(floor_labels, rotation=45)
            axes[idx].set_yticklabels(floor_labels)
            axes[idx].set_title(f'{floor} Similarity Matrix')
            axes[idx].set_xlabel('Track ID')
            axes[idx].set_ylabel('Track ID')
            
            # 격자 추가
            axes[idx].grid(True, alpha=0.3)
        
        # 전체 컬러바 추가
        fig.colorbar(im, ax=axes, shrink=0.8, label='Cosine Similarity')
        
        plt.tight_layout()
        
        # 저장
        floor_heatmap_file = output_path / 'similarity_matrix_by_floor.png'
        plt.savefig(floor_heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"층별 유사도 행렬 저장: {floor_heatmap_file}")
        
        # 3. 유사도 분포 히스토그램
        plt.figure(figsize=(12, 8))
        
        # 상삼각 행렬만 사용 (중복 제거)
        upper_tri_indices = np.triu_indices_from(similarity_matrix, k=1)
        similarities = similarity_matrix[upper_tri_indices]
        
        # 히스토그램
        plt.hist(similarities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(self.similarity_threshold, color='red', linestyle='--', 
                   label=f'Threshold: {self.similarity_threshold}')
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('Distribution of Cosine Similarities')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 통계 정보 추가
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        plt.text(0.7, 0.8, f'Mean: {mean_sim:.3f}\nStd: {std_sim:.3f}', 
                transform=plt.gca().transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
        
        plt.tight_layout()
        
        # 저장
        histogram_file = output_path / 'similarity_distribution.png'
        plt.savefig(histogram_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"유사도 분포 히스토그램 저장: {histogram_file}")
        
        # 4. 매칭 결과 시각화 (유사도 기반)
        self._visualize_matching_results(similarity_matrix, all_ids, output_path)
        
        self.logger.info("유사도 행렬 시각화 완료!")
    
    def _visualize_matching_results(self, similarity_matrix: np.ndarray, all_ids: List[Tuple[str, int]], output_path: Path):
        """매칭 결과를 유사도와 함께 시각화"""
        plt.figure(figsize=(15, 10))
        
        # 매칭된 그룹들을 색상으로 구분
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.matches)))
        
        # 각 그룹별로 점 찍기
        for group_id, tracks in self.matches.items():
            if len(tracks) <= 1:
                continue  # 단일 track 그룹은 제외
            
            # 그룹 내 track들의 인덱스 찾기
            group_indices = []
            for floor, track_id in tracks:
                for i, (f, t) in enumerate(all_ids):
                    if f == floor and t == track_id:
                        group_indices.append(i)
                        break
            
            # 그룹 내 유사도들 시각화
            for i in range(len(group_indices)):
                for j in range(i+1, len(group_indices)):
                    idx1, idx2 = group_indices[i], group_indices[j]
                    similarity = similarity_matrix[idx1, idx2]
                    
                    # 점 크기는 유사도에 비례
                    size = max(50, similarity * 200)
                    
                    plt.scatter(idx1, idx2, s=size, c=[colors[group_id]], 
                              alpha=0.7, edgecolors='black', linewidth=0.5)
                    
                    # 유사도 값 표시
                    plt.text(idx1, idx2, f'{similarity:.2f}', 
                            ha='center', va='center', fontsize=8, fontweight='bold')
        
        # 축 설정
        labels = [f"{floor}_{track_id:03d}" for floor, track_id in all_ids]
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(range(len(labels)), labels)
        plt.xlabel('Track ID')
        plt.ylabel('Track ID')
        plt.title('Matching Results Visualization\n(Size ∝ Similarity)')
        plt.grid(True, alpha=0.3)
        
        # 범례 추가
        legend_elements = []
        for group_id, tracks in self.matches.items():
            if len(tracks) > 1:
                legend_elements.append(plt.scatter([], [], c=[colors[group_id]], 
                                                 label=f'Group {group_id} ({len(tracks)} tracks)'))
        
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        # 저장
        matching_viz_file = output_path / 'matching_results_visualization.png'
        plt.savefig(matching_viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"매칭 결과 시각화 저장: {matching_viz_file}")
    
    def run_matching_pipeline(self, output_dir: str):
        """
        전체 매칭 파이프라인 실행
        
        Args:
            output_dir: 출력 디렉토리
        """
        self.logger.info("매칭 파이프라인 시작...")
        
        # 1. Tracklets 데이터 로드
        self.load_tracklets()
        
        # 2. 임베딩 추출
        self.extract_embeddings()
        
        # 3. 유사도 행렬 계산
        similarity_matrix, all_ids = self.compute_similarity_matrix()
        
        # 4. 유사도 행렬 시각화
        self.visualize_similarity_matrix(similarity_matrix, all_ids, output_dir)
        
        # 5. 매칭 찾기
        self.find_matches(similarity_matrix, all_ids)
        
        # 6. 통합 ID 매핑 생성
        self.create_unified_mapping()
        
        # 7. 결과 저장
        self.save_results(output_dir)
        
        # 8. 시각화
        self.visualize_matches(output_dir)
        
        # 9. 통합 ID별 이미지 복사
        self.copy_images_by_unified_id(output_dir)
        
        # 10. 매칭되지 않은 track 이미지 복사
        self.copy_unmatched_images(output_dir)
        
        self.logger.info("매칭 파이프라인 완료!")


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='카메라 간 사람 매칭 시스템')
    parser.add_argument('--tracklets_dir', type=str, 
                       default='/data/reid/reid_master/tracklets',
                       help='Tracklets 디렉토리 경로')
    parser.add_argument('--model_path', type=str,
                       default='/data/reid/reid_master/reid_embedding_extractor/checkpoints/swin_base_msmt17.pth',
                       help='ReID 모델 경로')
    parser.add_argument('--config_path', type=str,
                       default='/data/reid/reid_master/reid_embedding_extractor/models/solider/configs/msmt17/swin_base.yml',
                       help='모델 설정 파일 경로')
    parser.add_argument('--output_dir', type=str,
                       default='/data/reid/reid_master/cross_camera_results',
                       help='출력 디렉토리')
    parser.add_argument('--similarity_threshold', type=float, default=0.7,
                       help='유사도 임계값')
    parser.add_argument('--device', type=str, default='cuda',
                       help='사용할 디바이스 (cpu/cuda)')
    
    args = parser.parse_args()
    
    # 매칭 시스템 초기화
    matcher = CrossCameraMatcher(
        tracklets_dir=args.tracklets_dir,
        model_path=args.model_path,
        config_path=args.config_path,
        similarity_threshold=args.similarity_threshold,
        device=args.device
    )
    
    # 매칭 파이프라인 실행
    matcher.run_matching_pipeline(args.output_dir)


if __name__ == "__main__":
    main()
