#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
독립적인 설정 시스템 - SOLIDER_REID 의존성 제거
"""

import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import yaml


@dataclass
class ModelConfig:
    """모델 설정"""
    name: str = 'transformer'
    transformer_type: str = 'swin_base_patch4_window7_224'
    semantic_weight: float = 0.2
    stride_size: list = field(default_factory=lambda: [16, 16])
    pretrain_hw_ratio: int = 2
    device_id: str = '0'
    
    # 손실 함수 설정
    metric_loss_type: str = 'triplet'
    if_labelsmooth: str = 'off'
    if_with_center: str = 'no'
    no_margin: bool = True
    
    # 드롭아웃 설정
    dropout_rate: float = 0.0
    drop_path: float = 0.1
    drop_out: float = 0.0
    att_drop_rate: float = 0.0


@dataclass
class InputConfig:
    """입력 설정"""
    size_train: list = field(default_factory=lambda: [384, 128])
    size_test: list = field(default_factory=lambda: [384, 128])
    prob: float = 0.5  # random horizontal flip
    re_prob: float = 0.5  # random erasing
    padding: int = 10
    pixel_mean: list = field(default_factory=lambda: [0.5, 0.5, 0.5])
    pixel_std: list = field(default_factory=lambda: [0.5, 0.5, 0.5])


@dataclass
class TestConfig:
    """테스트 설정"""
    eval: bool = True
    ims_per_batch: int = 256
    re_ranking: bool = False
    weight: str = ''
    neck_feat: str = 'before'
    feat_norm: str = 'yes'
    dist_mat: str = 'dist_mat.npy'


class Config:
    """독립적인 설정 클래스"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.model = ModelConfig()
        self.input = InputConfig()
        self.test = TestConfig()
        
        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)
    
    def load_from_file(self, config_path: str):
        """YAML 파일에서 설정 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        # 모델 설정
        if 'MODEL' in config_dict:
            model_config = config_dict['MODEL']
            for key, value in model_config.items():
                # 대문자 키를 소문자로 변환하여 매핑
                attr_name = key.lower()
                if hasattr(self.model, attr_name):
                    setattr(self.model, attr_name, value)
                # 특별한 매핑 처리
                elif key == 'TRANSFORMER_TYPE':
                    self.model.transformer_type = value
                elif key == 'STRIDE_SIZE':
                    self.model.stride_size = value
                elif key == 'PRETRAIN_HW_RATIO':
                    self.model.pretrain_hw_ratio = value
                elif key == 'METRIC_LOSS_TYPE':
                    self.model.metric_loss_type = value
                elif key == 'IF_LABELSMOOTH':
                    self.model.if_labelsmooth = value
                elif key == 'IF_WITH_CENTER':
                    self.model.if_with_center = value
                elif key == 'NO_MARGIN':
                    self.model.no_margin = value
                elif key == 'DEVICE_ID':
                    self.model.device_id = value
        
        # 입력 설정
        if 'INPUT' in config_dict:
            input_config = config_dict['INPUT']
            for key, value in input_config.items():
                attr_name = key.lower()
                if hasattr(self.input, attr_name):
                    setattr(self.input, attr_name, value)
                # 특별한 매핑 처리
                elif key == 'SIZE_TRAIN':
                    self.input.size_train = value
                elif key == 'SIZE_TEST':
                    self.input.size_test = value
                elif key == 'PIXEL_MEAN':
                    self.input.pixel_mean = value
                elif key == 'PIXEL_STD':
                    self.input.pixel_std = value
                elif key == 'PROB':
                    self.input.prob = value
                elif key == 'RE_PROB':
                    self.input.re_prob = value
                elif key == 'PADDING':
                    self.input.padding = value
        
        # 테스트 설정
        if 'TEST' in config_dict:
            test_config = config_dict['TEST']
            for key, value in test_config.items():
                attr_name = key.lower()
                if hasattr(self.test, attr_name):
                    setattr(self.test, attr_name, value)
                # 특별한 매핑 처리
                elif key == 'IMS_PER_BATCH':
                    self.test.ims_per_batch = value
                elif key == 'RE_RANKING':
                    self.test.re_ranking = value
                elif key == 'NECK_FEAT':
                    self.test.neck_feat = value
                elif key == 'FEAT_NORM':
                    self.test.feat_norm = value
    
    def merge_from_file(self, config_path: str):
        """기존 설정에 파일 설정 병합"""
        self.load_from_file(config_path)
    
    def merge_from_list(self, opts: list):
        """명령행 인자로 설정 병합"""
        for i in range(0, len(opts), 2):
            if i + 1 < len(opts):
                key = opts[i]
                value = opts[i + 1]
                
                # 키를 파싱하여 적절한 섹션에 설정
                if key.startswith('MODEL.'):
                    attr_name = key[6:].lower()
                    if hasattr(self.model, attr_name):
                        # 타입 변환
                        if isinstance(getattr(self.model, attr_name), bool):
                            value = value.lower() in ('true', '1', 'yes', 'on')
                        elif isinstance(getattr(self.model, attr_name), int):
                            value = int(value)
                        elif isinstance(getattr(self.model, attr_name), float):
                            value = float(value)
                        elif isinstance(getattr(self.model, attr_name), list):
                            value = eval(value) if isinstance(value, str) else value
                        setattr(self.model, attr_name, value)
                elif key.startswith('INPUT.'):
                    attr_name = key[6:].lower()
                    if hasattr(self.input, attr_name):
                        if isinstance(getattr(self.input, attr_name), bool):
                            value = value.lower() in ('true', '1', 'yes', 'on')
                        elif isinstance(getattr(self.input, attr_name), int):
                            value = int(value)
                        elif isinstance(getattr(self.input, attr_name), float):
                            value = float(value)
                        elif isinstance(getattr(self.input, attr_name), list):
                            value = eval(value) if isinstance(value, str) else value
                        setattr(self.input, attr_name, value)
                elif key.startswith('TEST.'):
                    attr_name = key[5:].lower()
                    if hasattr(self.test, attr_name):
                        if isinstance(getattr(self.test, attr_name), bool):
                            value = value.lower() in ('true', '1', 'yes', 'on')
                        elif isinstance(getattr(self.test, attr_name), int):
                            value = int(value)
                        elif isinstance(getattr(self.test, attr_name), float):
                            value = float(value)
                        elif isinstance(getattr(self.test, attr_name), list):
                            value = eval(value) if isinstance(value, str) else value
                        setattr(self.test, attr_name, value)
    
    def freeze(self):
        """설정을 불변으로 만들기 (호환성을 위해 유지)"""
        pass
    
    def __getattr__(self, name):
        """속성 접근을 위한 매직 메서드"""
        if name == 'MODEL':
            return self.model
        elif name == 'INPUT':
            return self.input
        elif name == 'TEST':
            return self.test
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __getitem__(self, key):
        """딕셔너리 스타일 접근을 위한 매직 메서드"""
        if key == 'MODEL':
            return self.model
        elif key == 'INPUT':
            return self.input
        elif key == 'TEST':
            return self.test
        else:
            raise KeyError(f"'{key}' not found")
    
    def __str__(self):
        """설정을 문자열로 출력"""
        return f"""Config:
  MODEL:
    name: {self.model.name}
    transformer_type: {self.model.transformer_type}
    semantic_weight: {self.model.semantic_weight}
    stride_size: {self.model.stride_size}
    device_id: {self.model.device_id}
  
  INPUT:
    size_train: {self.input.size_train}
    size_test: {self.input.size_test}
    pixel_mean: {self.input.pixel_mean}
    pixel_std: {self.input.pixel_std}
  
  TEST:
    eval: {self.test.eval}
    ims_per_batch: {self.test.ims_per_batch}
    neck_feat: {self.test.neck_feat}
    feat_norm: {self.test.feat_norm}
"""


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

TEST:
  EVAL: True
  IMS_PER_BATCH: 256
  RE_RANKING: False
  WEIGHT: ''
  NECK_FEAT: 'before'
  FEAT_NORM: 'yes'
"""
    return config_content


def find_config_file(model_name: str, dataset: str = 'msmt17') -> Optional[str]:
    """모델 이름에 따라 설정 파일을 자동으로 찾기"""
    # 여러 데이터셋에서 찾기
    datasets = [dataset, 'msmt17', 'market1501'] if dataset != 'msmt17' else ['msmt17', 'market1501']
    
    for ds in datasets:
        config_dir = os.path.join(os.path.dirname(__file__), 'configs', ds)
        
        # 가능한 설정 파일 이름들
        possible_names = [
            f"{model_name}.yml",
            f"{model_name}.yaml",
            f"{model_name}_config.yml",
            f"{model_name}_config.yaml"
        ]
        
        for name in possible_names:
            config_path = os.path.join(config_dir, name)
            if os.path.exists(config_path):
                return config_path
    
    return None


def get_available_configs(dataset: str = 'msmt17') -> List[str]:
    """사용 가능한 설정 파일 목록 반환"""
    if dataset == 'all':
        # 모든 데이터셋의 설정 파일 반환
        all_configs = []
        configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
        
        if os.path.exists(configs_dir):
            for ds in os.listdir(configs_dir):
                ds_path = os.path.join(configs_dir, ds)
                if os.path.isdir(ds_path):
                    for file in os.listdir(ds_path):
                        if file.endswith(('.yml', '.yaml')):
                            config_name = file.replace('.yml', '').replace('.yaml', '')
                            all_configs.append(f"{config_name} ({ds})")
        
        return sorted(all_configs)
    else:
        # 특정 데이터셋의 설정 파일만 반환
        config_dir = os.path.join(os.path.dirname(__file__), 'configs', dataset)
        
        if not os.path.exists(config_dir):
            return []
        
        configs = []
        for file in os.listdir(config_dir):
            if file.endswith(('.yml', '.yaml')):
                configs.append(file.replace('.yml', '').replace('.yaml', ''))
        
        return sorted(configs)


def get_available_datasets() -> List[str]:
    """사용 가능한 데이터셋 목록 반환"""
    configs_dir = os.path.join(os.path.dirname(__file__), 'configs')
    
    if not os.path.exists(configs_dir):
        return []
    
    datasets = []
    for item in os.listdir(configs_dir):
        item_path = os.path.join(configs_dir, item)
        if os.path.isdir(item_path):
            datasets.append(item)
    
    return sorted(datasets)


# 전역 설정 인스턴스
cfg = Config()