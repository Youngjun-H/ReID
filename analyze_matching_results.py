#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
매칭 결과 분석 및 검증 도구
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict, Counter
import argparse


class MatchingAnalyzer:
    """매칭 결과 분석 클래스"""
    
    def __init__(self, results_dir: str):
        """
        초기화
        
        Args:
            results_dir: 결과 디렉토리 경로
        """
        self.results_dir = Path(results_dir)
        
        # 결과 파일들 로드
        self.load_results()
    
    def load_results(self):
        """결과 파일들 로드"""
        # 통합 ID 매핑 로드
        mapping_file = self.results_dir / 'unified_id_mapping.json'
        if mapping_file.exists():
            with open(mapping_file, 'r', encoding='utf-8') as f:
                self.unified_ids = json.load(f)
        else:
            self.unified_ids = {}
        
        # 매칭 결과 로드
        matches_file = self.results_dir / 'matches.json'
        if matches_file.exists():
            with open(matches_file, 'r', encoding='utf-8') as f:
                self.matches = json.load(f)
        else:
            self.matches = {}
        
        # 통계 정보 로드
        stats_file = self.results_dir / 'statistics.json'
        if stats_file.exists():
            with open(stats_file, 'r', encoding='utf-8') as f:
                self.statistics = json.load(f)
        else:
            self.statistics = {}
    
    def analyze_matching_quality(self) -> Dict:
        """매칭 품질 분석"""
        print("=== 매칭 품질 분석 ===")
        
        # 기본 통계
        total_tracks = self.statistics.get('total_tracks', 0)
        total_unified_ids = self.statistics.get('total_unified_ids', 0)
        
        print(f"총 Track 수: {total_tracks}")
        print(f"통합 ID 수: {total_unified_ids}")
        print(f"압축률: {total_unified_ids / total_tracks * 100:.1f}%")
        
        # 층별 분석
        print("\n=== 층별 분석 ===")
        for floor in ['1F', '2F', '3F']:
            if floor in self.statistics.get('tracks_per_floor', {}):
                tracks = self.statistics['tracks_per_floor'][floor]
                unified = self.statistics.get('unified_ids_per_floor', {}).get(floor, 0)
                print(f"{floor}: {tracks} tracks → {unified} unified IDs")
        
        # 매칭 그룹 분석
        print("\n=== 매칭 그룹 분석 ===")
        group_sizes = []
        multi_floor_groups = 0
        
        for group_id, tracks in self.matches.items():
            group_size = len(tracks)
            group_sizes.append(group_size)
            
            # 다층 매칭 그룹 확인
            floors = set(track[0] for track in tracks)
            if len(floors) > 1:
                multi_floor_groups += 1
        
        if group_sizes:
            print(f"그룹 크기 분포:")
            print(f"  - 평균: {np.mean(group_sizes):.2f}")
            print(f"  - 중앙값: {np.median(group_sizes):.2f}")
            print(f"  - 최대: {max(group_sizes)}")
            print(f"  - 최소: {min(group_sizes)}")
            print(f"  - 다층 매칭 그룹: {multi_floor_groups}개")
        
        return {
            'total_tracks': total_tracks,
            'total_unified_ids': total_unified_ids,
            'compression_ratio': total_unified_ids / total_tracks if total_tracks > 0 else 0,
            'group_sizes': group_sizes,
            'multi_floor_groups': multi_floor_groups
        }
    
    def visualize_matching_distribution(self, output_dir: str):
        """매칭 분포 시각화"""
        print("=== 매칭 분포 시각화 ===")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 층별 Track 분포
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 층별 Track 수
        floors = ['1F', '2F', '3F']
        track_counts = [self.statistics.get('tracks_per_floor', {}).get(floor, 0) for floor in floors]
        
        axes[0, 0].bar(floors, track_counts, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 0].set_title('층별 Track 수')
        axes[0, 0].set_ylabel('Track 수')
        
        # 층별 통합 ID 수
        unified_counts = [self.statistics.get('unified_ids_per_floor', {}).get(floor, 0) for floor in floors]
        
        axes[0, 1].bar(floors, unified_counts, color=['skyblue', 'lightgreen', 'lightcoral'])
        axes[0, 1].set_title('층별 통합 ID 수')
        axes[0, 1].set_ylabel('통합 ID 수')
        
        # 그룹 크기 분포
        group_sizes = []
        for group_id, tracks in self.matches.items():
            group_sizes.append(len(tracks))
        
        if group_sizes:
            axes[1, 0].hist(group_sizes, bins=range(1, max(group_sizes) + 2), 
                           color='lightblue', edgecolor='black', alpha=0.7)
            axes[1, 0].set_title('그룹 크기 분포')
            axes[1, 0].set_xlabel('그룹 크기')
            axes[1, 0].set_ylabel('빈도')
        
        # 층 간 매칭 히트맵
        floor_matrix = np.zeros((3, 3))
        floor_names = ['1F', '2F', '3F']
        
        for group_id, tracks in self.matches.items():
            if len(tracks) > 1:  # 다중 track 그룹만
                floors_in_group = [track[0] for track in tracks]
                for i, floor1 in enumerate(floor_names):
                    for j, floor2 in enumerate(floor_names):
                        if floor1 in floors_in_group and floor2 in floors_in_group:
                            floor_matrix[i, j] += 1
        
        im = axes[1, 1].imshow(floor_matrix, cmap='YlOrRd', aspect='auto')
        axes[1, 1].set_xticks(range(3))
        axes[1, 1].set_yticks(range(3))
        axes[1, 1].set_xticklabels(floor_names)
        axes[1, 1].set_yticklabels(floor_names)
        axes[1, 1].set_title('층 간 매칭 히트맵')
        
        # 히트맵에 값 표시
        for i in range(3):
            for j in range(3):
                text = axes[1, 1].text(j, i, f'{int(floor_matrix[i, j])}',
                                     ha="center", va="center", color="black")
        
        plt.tight_layout()
        plt.savefig(output_path / 'matching_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"시각화 완료: {output_path / 'matching_distribution.png'}")
    
    def generate_detailed_report(self, output_dir: str):
        """상세 보고서 생성"""
        print("=== 상세 보고서 생성 ===")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. 매칭 품질 분석
        quality_analysis = self.analyze_matching_quality()
        
        # 2. 층별 상세 정보
        floor_details = {}
        for floor in ['1F', '2F', '3F']:
            if floor in self.unified_ids:
                floor_details[floor] = {
                    'total_tracks': len(self.unified_ids[floor]),
                    'unique_unified_ids': len(set(self.unified_ids[floor].values())),
                    'mapping': self.unified_ids[floor]
                }
        
        # 3. 매칭 그룹 상세 정보
        group_details = {}
        for group_id, tracks in self.matches.items():
            floors = [track[0] for track in tracks]
            group_details[group_id] = {
                'size': len(tracks),
                'floors': list(set(floors)),
                'tracks': tracks,
                'is_multi_floor': len(set(floors)) > 1
            }
        
        # 4. 보고서 생성
        report = {
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'quality_analysis': quality_analysis,
            'floor_details': floor_details,
            'group_details': group_details,
            'statistics': self.statistics
        }
        
        # JSON 보고서 저장
        report_file = output_path / 'detailed_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # CSV 보고서 생성
        self._generate_csv_reports(output_path)
        
        print(f"상세 보고서 생성 완료: {output_path}")
    
    def _generate_csv_reports(self, output_path: Path):
        """CSV 보고서 생성"""
        # 1. 층별 매핑 테이블
        mapping_data = []
        for floor, mappings in self.unified_ids.items():
            for track_id, unified_id in mappings.items():
                mapping_data.append({
                    'floor': floor,
                    'original_track_id': track_id,
                    'unified_id': unified_id
                })
        
        if mapping_data:
            mapping_df = pd.DataFrame(mapping_data)
            mapping_df.to_csv(output_path / 'id_mapping.csv', index=False)
        
        # 2. 그룹 정보 테이블
        group_data = []
        for group_id, tracks in self.matches.items():
            floors = [track[0] for track in tracks]
            group_data.append({
                'group_id': group_id,
                'size': len(tracks),
                'floors': ', '.join(set(floors)),
                'is_multi_floor': len(set(floors)) > 1,
                'tracks': ', '.join([f"{t[0]}-{t[1]}" for t in tracks])
            })
        
        if group_data:
            group_df = pd.DataFrame(group_data)
            group_df.to_csv(output_path / 'group_info.csv', index=False)
    
    def find_potential_issues(self) -> Dict:
        """잠재적 문제점 찾기"""
        print("=== 잠재적 문제점 분석 ===")
        
        issues = {
            'large_groups': [],
            'single_track_groups': [],
            'inconsistent_mappings': [],
            'missing_floors': []
        }
        
        # 1. 큰 그룹 찾기 (5개 이상)
        for group_id, tracks in self.matches.items():
            if len(tracks) >= 5:
                issues['large_groups'].append({
                    'group_id': group_id,
                    'size': len(tracks),
                    'tracks': tracks
                })
        
        # 2. 단일 track 그룹들
        for group_id, tracks in self.matches.items():
            if len(tracks) == 1:
                issues['single_track_groups'].append({
                    'group_id': group_id,
                    'track': tracks[0]
                })
        
        # 3. 층별 일관성 확인
        for floor in ['1F', '2F', '3F']:
            if floor not in self.unified_ids:
                issues['missing_floors'].append(floor)
        
        # 결과 출력
        if issues['large_groups']:
            print(f"큰 그룹 ({len(issues['large_groups'])}개): 5개 이상 track을 포함한 그룹")
        
        if issues['single_track_groups']:
            print(f"단일 track 그룹 ({len(issues['single_track_groups'])}개): 매칭되지 않은 track들")
        
        if issues['missing_floors']:
            print(f"누락된 층: {issues['missing_floors']}")
        
        return issues


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='매칭 결과 분석 도구')
    parser.add_argument('--results_dir', type=str,
                       default='/data/reid/reid_master/cross_camera_results',
                       help='결과 디렉토리 경로')
    parser.add_argument('--output_dir', type=str,
                       default='/data/reid/reid_master/analysis_results',
                       help='분석 결과 출력 디렉토리')
    
    args = parser.parse_args()
    
    # 분석기 초기화
    analyzer = MatchingAnalyzer(args.results_dir)
    
    # 분석 실행
    print("매칭 결과 분석 시작...")
    
    # 1. 매칭 품질 분석
    analyzer.analyze_matching_quality()
    
    # 2. 시각화
    analyzer.visualize_matching_distribution(args.output_dir)
    
    # 3. 상세 보고서 생성
    analyzer.generate_detailed_report(args.output_dir)
    
    # 4. 잠재적 문제점 분석
    analyzer.find_potential_issues()
    
    print("\n분석 완료!")


if __name__ == "__main__":
    main()
