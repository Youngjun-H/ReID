#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSNet 기반 Cross-Camera Matching 파이프라인
- 각 층별 track_* 디렉토리에서 가운데 프레임 이미지를 대표로 선택
- OSNet 임베딩 추출 후 코사인 유사도 0.8 이상을 하나의 글로벌 ID로 그룹화
- 결과를 JSON으로 저장
"""

import os
import json
from pathlib import Path
from typing import Dict, List
import argparse
import logging
import numpy as np

from reid_dataset_builder.features.models import OSNetFeatureExtractor
from reid_dataset_builder.matching import CrossCameraMatcher
from reid_dataset_builder.matching.results_utils import (
    copy_images_by_unified_id,
    create_unified_images_summary,
    visualize_groups,
)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger('osnet_tracklet_matching')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def find_image_files_in_dir(directory: Path, exts: List[str] = None) -> List[Path]:
    if exts is None:
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
    files: List[Path] = []
    for ext in exts:
        files.extend(sorted(directory.glob(f'*{ext}')))
        files.extend(sorted(directory.glob(f'*{ext.upper()}')))
    return files


def collect_representative_images(tracklets_dir: Path) -> Dict[str, Dict[int, Path]]:
    """
    층별/트랙별 대표 프레임(가운데) 이미지 경로 수집

    Returns:
        {floor: {track_id: image_path}}
    """
    results: Dict[str, Dict[int, Path]] = {}

    if not tracklets_dir.exists():
        raise FileNotFoundError(f'Tracklets dir not found: {tracklets_dir}')

    for floor_dir in sorted([d for d in tracklets_dir.iterdir() if d.is_dir()]):
        floor = floor_dir.name
        results[floor] = {}

        for track_dir in sorted([d for d in floor_dir.iterdir() if d.is_dir() and d.name.startswith('track_') or d.name.startswith('id_')]):
            # track_#### or id_#### both supported
            try:
                # extract numeric id
                parts = track_dir.name.split('_')
                track_id = int(parts[1])
            except Exception:
                continue

            images = find_image_files_in_dir(track_dir)
            if not images:
                # fallback: look recursively (in case nested)
                images = sorted([p for p in track_dir.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
            if not images:
                continue

            mid_idx = len(images) // 2
            results[floor][track_id] = images[mid_idx]

    return results


def collect_all_images(tracklets_dir: Path) -> Dict[str, Dict[int, List[Path]]]:
    """
    층별/트랙별 모든 이미지 경로 수집

    Returns:
        {floor: {track_id: [image_path, ...]}}
    """
    results: Dict[str, Dict[int, List[Path]]] = {}
    if not tracklets_dir.exists():
        raise FileNotFoundError(f'Tracklets dir not found: {tracklets_dir}')

    for floor_dir in sorted([d for d in tracklets_dir.iterdir() if d.is_dir()]):
        floor = floor_dir.name
        results[floor] = {}
        for track_dir in sorted([d for d in floor_dir.iterdir() if d.is_dir() and d.name.startswith('track_') or d.name.startswith('id_')]):
            try:
                parts = track_dir.name.split('_')
                track_id = int(parts[1])
            except Exception:
                continue
            images = find_image_files_in_dir(track_dir)
            if not images:
                images = sorted([p for p in track_dir.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}])
            if not images:
                continue
            results[floor][track_id] = images
    return results


def extract_embeddings_osnet(representatives: Dict[str, Dict[int, Path]], model_path: str, device: str = 'cuda') -> Dict[str, Dict[int, np.ndarray]]:
    extractor = OSNetFeatureExtractor(
        model_name='osnet_ain_x1_0',
        model_path=model_path,
        device=device,
    )

    embeddings: Dict[str, Dict[int, np.ndarray]] = {}
    for floor, tracks in representatives.items():
        embeddings[floor] = {}
        for track_id, img_path in tracks.items():
            feats = extractor(str(img_path))  # (1, D) torch.Tensor
            emb = feats.squeeze(0).detach().cpu().numpy()
            embeddings[floor][track_id] = emb
    return embeddings


def save_results(output_dir: Path, similarity_matrix: np.ndarray, all_ids: List, matches: Dict, unified_ids: Dict[str, Dict[int, int]]):
    output_dir.mkdir(parents=True, exist_ok=True)

    # 매칭/매핑 저장
    with open(output_dir / 'matches.json', 'w', encoding='utf-8') as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)
    with open(output_dir / 'unified_id_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(unified_ids, f, indent=2, ensure_ascii=False)

    # ID 인덱스와 라벨 저장
    ids_serializable = [(floor, int(tid)) for floor, tid in all_ids]
    with open(output_dir / 'all_ids.json', 'w', encoding='utf-8') as f:
        json.dump(ids_serializable, f, indent=2, ensure_ascii=False)

    # 유사도 행렬 저장 (numpy)
    np.save(output_dir / 'similarity_matrix.npy', similarity_matrix)


def main():
    parser = argparse.ArgumentParser(description='OSNet 기반 Cross-Camera Matching')
    parser.add_argument('--tracklets_dir', type=str, default='/data/reid/reid_master/tracklets')
    parser.add_argument('--model_path', type=str, default='/data/reid/reid_master/reid_dataset_builder/features/models/osnet/checkpoints/osnet_ain_x1_0_msmt17.pth')
    parser.add_argument('--output_dir', type=str, default='/data/reid/reid_master/osnet_cross_camera_results_0.5_new')
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'euclidean'],
                        help='유사도/거리 지표 선택 (cosine: 높을수록 매칭, euclidean: 낮을수록 매칭)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='임계값 (cosine: 최소 유사도, euclidean: 최대 거리)')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    logger = setup_logger()
    tracklets_dir = Path(args.tracklets_dir)
    output_dir = Path(args.output_dir)

    logger.info('대표 프레임 수집 중...')
    reps = collect_representative_images(tracklets_dir)
    logger.info('트랙 전체 이미지 수집 중...')
    all_images = collect_all_images(tracklets_dir)
    total_tracks = sum(len(v) for v in reps.values())
    logger.info(f'수집된 트랙 수: {total_tracks}')

    if total_tracks == 0:
        logger.error('대표 프레임을 찾지 못했습니다. 디렉토리 구조를 확인하세요.')
        return

    logger.info('OSNet 임베딩 추출 중...')
    embeddings_by_floor = extract_embeddings_osnet(reps, args.model_path, device=args.device)

    logger.info('매칭 수행 중...')
    matcher = CrossCameraMatcher(threshold=args.threshold, metric=args.metric)
    mat, all_ids, matches, unified = matcher.match_from_embeddings_dict(embeddings_by_floor)

    logger.info('결과 저장 중...')
    save_results(output_dir, mat, all_ids, matches, unified)

    logger.info('통합 ID별 이미지 복사 중...')
    unified_images_dir = copy_images_by_unified_id(all_images, matches, output_dir)

    logger.info('요약 생성 중...')
    create_unified_images_summary(all_images, matches, unified_images_dir)

    logger.info('간단 시각화 생성 중...')
    visualize_groups(reps, matches, output_dir, max_groups=12)
    logger.info(f'완료: {output_dir}')


if __name__ == '__main__':
    main()
