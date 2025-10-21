import time
import os
from tqdm import tqdm


class ProgressVisualizer:
    """비디오 처리 진행상황을 시각화하는 클래스"""
    
    def __init__(self, video_path, total_frames, fps, conf, out_dir, cam_id, frame_interval=1):
        self.video_path = video_path
        self.total_frames = total_frames
        self.fps = fps
        self.conf = conf
        self.out_dir = out_dir
        self.cam_id = cam_id
        self.frame_interval = frame_interval
        
        # 통계 변수
        self.frame_id = 0
        self.total_tracks = 0
        self.unique_ids = set()
        self.start_time = time.time()
        
        # 동적 계산 모드 여부
        self.dynamic_mode = total_frames == -1
        
        # Progress bar 초기화
        self.pbar = None
        self._init_progress_bar()
    
    def _init_progress_bar(self):
        """Progress bar 초기화"""
        video_extension = os.path.splitext(self.video_path)[1].lower()
        is_avi = video_extension == '.avi'
        
        # frame_interval에 따른 설명 생성
        interval_desc = f" (매 {self.frame_interval}프레임)" if self.frame_interval > 1 else ""
        
        if self.dynamic_mode:
            if is_avi:
                # AVI 동적 모드: AVI 파일 특성 명시
                self.pbar = tqdm(desc=f"🎥 AVI 처리 중{interval_desc}", 
                                unit="frame", unit_scale=True)
            else:
                # 일반 동적 모드: 총 프레임 수를 모르므로 무한 진행률 표시
                self.pbar = tqdm(desc=f"🎥 처리 중{interval_desc}", 
                                unit="frame", unit_scale=True)
        else:
            # 정적 모드: 알려진 총 프레임 수 사용
            self.pbar = tqdm(total=self.total_frames, desc=f"🎥 처리 중{interval_desc}", 
                            unit="frame", unit_scale=True)
    
    def print_video_info(self):
        """비디오 정보 출력"""
        video_extension = os.path.splitext(self.video_path)[1].lower()
        is_avi = video_extension == '.avi'
        
        if self.dynamic_mode:
            if is_avi:
                duration_info = "알 수 없음 (AVI 동적 모드)"
            else:
                duration_info = "알 수 없음 (동적 계산 모드)"
        else:
            duration = self.total_frames / self.fps if self.fps > 0 else 0
            duration_info = f"{duration/60:.1f}분"
        
        print(f"🎬 비디오 정보:")
        print(f"   📁 파일: {os.path.basename(self.video_path)}")
        if self.dynamic_mode:
            if is_avi:
                print(f"   🎞️  총 프레임: 알 수 없음 (AVI 동적 계산)")
            else:
                print(f"   🎞️  총 프레임: 알 수 없음 (동적 계산)")
        else:
            print(f"   🎞️  총 프레임: {self.total_frames:,}")
        print(f"   ⏱️  FPS: {self.fps:.1f}")
        print(f"   ⏰ 길이: {duration_info}")
        print(f"   🎯 신뢰도: {self.conf}")
        print(f"   🔄 프레임 간격: {self.frame_interval} (매 {self.frame_interval}프레임마다 처리)")
        print(f"   📤 출력: {self.out_dir}/{self.cam_id}")
        if is_avi and self.dynamic_mode:
            print(f"   💡 AVI 파일: 실시간 프레임 수로 진행률 표시")
        if self.frame_interval > 1:
            print(f"   ⚡ 성능 최적화: {self.frame_interval}배 빠른 처리")
        print("-" * 50)
    
    def update_progress(self, tracks, frame_count=None, processed_frames=None):
        """진행상황 업데이트"""
        self.frame_id += 1
        self.total_tracks += len(tracks)
        
        # 고유 ID 수집
        for track in tracks:
            if len(track) >= 5:  # (x1, y1, x2, y2, tid, conf)
                self.unique_ids.add(int(track[4]))
        
        # Progress bar 업데이트 (매 10프레임마다)
        if self.frame_id % 10 == 0:
            if self.dynamic_mode:
                # 동적 모드: 프레임 수만 업데이트
                self.pbar.update(10)
                
                # frame_count와 processed_frames가 제공된 경우 사용
                if frame_count is not None and processed_frames is not None:
                    self.pbar.set_postfix({
                        'Total Frames': f'{frame_count:,}',
                        'Processed': f'{processed_frames:,}',
                        'Current': len(tracks),  # 현재 프레임의 tracklet 수
                        'Total Tracks': f'{self.total_tracks:,}',  # 누적된 총 tracklet 수
                        'Unique IDs': len(self.unique_ids)
                    })
                else:
                    self.pbar.set_postfix({
                        'Frames': f'{self.frame_id:,}',
                        'Current': len(tracks),  # 현재 프레임의 tracklet 수
                        'Total Tracks': f'{self.total_tracks:,}',  # 누적된 총 tracklet 수
                        'Unique IDs': len(self.unique_ids)
                    })
            else:
                # 정적 모드: 진행률 포함
                self.pbar.update(10)
                progress_percent = (self.frame_id / self.total_frames) * 100 if self.total_frames > 0 else 0
                self.pbar.set_postfix({
                    'Current': len(tracks),  # 현재 프레임의 tracklet 수
                    'Total Tracks': f'{self.total_tracks:,}',  # 누적된 총 tracklet 수
                    'Unique IDs': len(self.unique_ids),
                    'Progress': f'{progress_percent:.1f}%'
                })
    
    def close(self):
        """Progress bar 종료 및 최종 통계 출력"""
        if self.pbar:
            self.pbar.close()
        
        # 처리 시간 계산
        end_time = time.time()
        processing_time = end_time - self.start_time
        avg_fps = self.frame_id / processing_time if processing_time > 0 else 0
        
        # 최종 통계 출력
        print("\n" + "="*60)
        print("🎉 처리 완료!")
        print("="*60)
        print(f"📁 파일: {os.path.basename(self.video_path)}")
        print(f"🎞️  처리된 프레임: {self.frame_id:,}")
        print(f"⏱️  처리 시간: {processing_time:.1f}초 ({processing_time/60:.1f}분)")
        print(f"🚀 평균 FPS: {avg_fps:.1f}")
        print(f"👥 고유 ID 수: {len(self.unique_ids)}")
        print(f"📊 총 tracklet 수: {self.total_tracks:,}")
        print(f"📤 출력 위치: {self.out_dir}/{self.cam_id}")
        
        # ID별 통계
        if self.unique_ids:
            print(f"🆔 발견된 ID: {sorted(list(self.unique_ids))}")
        
        print("="*60)
    
    def get_frame_id(self):
        """현재 처리된 프레임 ID 반환 (frame_interval 고려)"""
        return self.frame_id
