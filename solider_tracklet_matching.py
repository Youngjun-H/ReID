#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SOLIDER 기반 Cross-Camera Matching 파이프라인
- 각 층별 track_* 디렉토리에서 가운데 프레임 이미지를 대표로 선택
- SOLIDER 임베딩 추출 후 선택한 metric(cosine/euclidean)과 threshold로 글로벌 ID 그룹화
- 결과를 JSON/NPY로 저장하고, 통합 ID별 이미지 복사/요약/시각화 수행
"""

import json
from pathlib import Path
from typing import Dict, List
import argparse
import logging
import numpy as np

from reid_dataset_builder.features.models import SOLIDEREmbeddingExtractor
from reid_dataset_builder.matching import CrossCameraMatcher
from reid_dataset_builder.matching.results_utils import (
    copy_images_by_unified_id,
    create_unified_images_summary,
    visualize_groups,
)


def setup_logger() -> logging.Logger:
    logger = logging.getLogger('solider_tracklet_matching')
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


def extract_embeddings_solider(
    representatives: Dict[str, Dict[int, Path]],
    model_path: str,
    config_path: str = None,
    device: str = 'cuda',
    semantic_weight: float = 0.2,
) -> Dict[str, Dict[int, np.ndarray]]:
    extractor = SOLIDEREmbeddingExtractor(
        model_path=model_path,
        config_path=config_path,
        device=device,
        semantic_weight=semantic_weight,
        image_size=(384, 128),
        normalize_features=True,
    )

    embeddings: Dict[str, Dict[int, np.ndarray]] = {}
    for floor, tracks in representatives.items():
        embeddings[floor] = {}
        for track_id, img_path in tracks.items():
            emb = extractor.extract_embedding(str(img_path))
            emb = np.asarray(emb)
            embeddings[floor][track_id] = emb
    return embeddings


def save_results(output_dir: Path, matrix: np.ndarray, all_ids: List, matches: Dict, unified_ids: Dict[str, Dict[int, int]]):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'matches.json', 'w', encoding='utf-8') as f:
        json.dump(matches, f, indent=2, ensure_ascii=False)
    with open(output_dir / 'unified_id_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(unified_ids, f, indent=2, ensure_ascii=False)

    ids_serializable = [(floor, int(tid)) for floor, tid in all_ids]
    with open(output_dir / 'all_ids.json', 'w', encoding='utf-8') as f:
        json.dump(ids_serializable, f, indent=2, ensure_ascii=False)

    np.save(output_dir / 'matrix.npy', matrix)


def main():
    parser = argparse.ArgumentParser(description='SOLIDER 기반 Cross-Camera Matching')
    parser.add_argument('--tracklets_dir', type=str, default='/data/reid/reid_master/tracklets')
    parser.add_argument('--model_path', type=str, default='/data/reid/reid_master/reid_dataset_builder/features/models/solider/checkpoints/swin_base_msmt17.pth')
    parser.add_argument('--config_path', type=str, default='/data/reid/reid_master/reid_dataset_builder/features/models/solider/configs/msmt17/swin_base.yml')
    parser.add_argument('--output_dir', type=str, default='/data/reid/reid_master/solider_cross_camera_results_0.7_semantic_weight_0.5')
    parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'euclidean'])
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--semantic_weight', type=float, default=0.5)
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

    logger.info('SOLIDER 임베딩 추출 중...')
    embeddings_by_floor = extract_embeddings_solider(
        reps,
        model_path=args.model_path,
        config_path=args.config_path,
        device=args.device,
        semantic_weight=args.semantic_weight,
    )

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