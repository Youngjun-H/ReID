#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image
import matplotlib.pyplot as plt


def copy_images_by_unified_id(
    track_images_by_floor: Dict[str, Dict[int, List[Path]]],
    matches: Dict[int, List[Tuple[str, int]]],
    output_dir: Path,
) -> Path:
    """
    통합 ID별로 원본 트랙 이미지들을 복사해 폴더 구조 생성
    """
    unified_images_dir = output_dir / 'unified_images'
    unified_images_dir.mkdir(parents=True, exist_ok=True)

    for group_id, tracks in matches.items():
        unified_id = group_id + 1
        group_dir = unified_images_dir / f'unified_id_{unified_id:03d}'
        group_dir.mkdir(parents=True, exist_ok=True)

        for floor, track_id in tracks:
            images = track_images_by_floor.get(floor, {}).get(track_id, [])
            for i, image_path in enumerate(images):
                dest_name = f"{floor}_{track_id:04d}_{i:04d}_{image_path.name}"
                dest_path = group_dir / dest_name
                try:
                    shutil.copy2(image_path, dest_path)
                except Exception:
                    # 실패해도 전체 파이프라인은 계속 진행
                    pass

        # 그룹 메타 저장
        group_info = {
            'unified_id': unified_id,
            'group_id': group_id,
            'tracks': tracks,
            'floors': sorted(list({f for f, _ in tracks})),
            'total_images': sum(len(track_images_by_floor.get(f, {}).get(t, [])) for f, t in tracks),
        }
        with open(group_dir / 'group_info.json', 'w', encoding='utf-8') as f:
            json.dump(group_info, f, indent=2, ensure_ascii=False)

    return unified_images_dir


def create_unified_images_summary(
    track_images_by_floor: Dict[str, Dict[int, List[Path]]],
    matches: Dict[int, List[Tuple[str, int]]],
    unified_images_dir: Path,
) -> Path:
    summary = {
        'total_unified_ids': len(matches),
        'total_tracks': sum(len(v) for v in track_images_by_floor.values()),
        'total_images': 0,
        'unified_ids': [],
    }

    for group_id, tracks in matches.items():
        unified_id = group_id + 1
        image_count = 0
        for floor, track_id in tracks:
            image_count += len(track_images_by_floor.get(floor, {}).get(track_id, []))

        summary['total_images'] += image_count
        summary['unified_ids'].append({
            'unified_id': unified_id,
            'group_id': group_id,
            'tracks': tracks,
            'image_count': image_count,
            'floors': sorted(list({f for f, _ in tracks})),
        })

    summary_file = unified_images_dir / 'summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary_file


def visualize_groups(
    representatives: Dict[str, Dict[int, Path]],
    matches: Dict[int, List[Tuple[str, int]]],
    output_dir: Path,
    max_groups: int = 12,
) -> None:
    """
    각 그룹의 대표 이미지를 가로로 나란히 배치한 썸네일 PNG 저장
    """
    vis_dir = output_dir / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 다중 트랙 그룹만 시각화
    multi_track = {gid: tr for gid, tr in matches.items() if len(tr) > 1}
    items = list(multi_track.items())[:max_groups]
    if not items:
        return

    for group_id, tracks in items:
        n = len(tracks)
        fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
        if n == 1:
            axes = [axes]

        for i, (floor, track_id) in enumerate(tracks):
            img_path = representatives.get(floor, {}).get(track_id)
            if img_path is None:
                axes[i].axis('off')
                continue
            try:
                img = Image.open(img_path)
                axes[i].imshow(img)
            except Exception:
                axes[i].axis('off')
                continue
            axes[i].set_title(f"{floor} {track_id:04d}")
            axes[i].axis('off')

        fig.suptitle(f"Group {group_id}  (Unified {group_id + 1})")
        fig.tight_layout()
        out_file = vis_dir / f'group_{group_id:03d}.png'
        try:
            fig.savefig(out_file, dpi=180, bbox_inches='tight')
        finally:
            plt.close(fig)


