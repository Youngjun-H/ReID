#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
독립적인 모델 팩토리 - SOLIDER_REID 의존성 제거
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging

try:
    # 패키지로 import될 때 (상대 import)
    from .models.swin_transformer import (
        swin_base_patch4_window7_224,
        swin_small_patch4_window7_224, 
        swin_tiny_patch4_window7_224
    )
except ImportError:
    # 직접 실행될 때 (절대 import)
    from models.swin_transformer import (
        swin_base_patch4_window7_224,
        swin_small_patch4_window7_224, 
        swin_tiny_patch4_window7_224
    )


def weights_init_xavier(m):
    """Xavier 초기화"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    """Kaiming 초기화"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    """분류기 초기화"""
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TransformerModel(nn.Module):
    """독립적인 Transformer 모델"""
    
    def __init__(self, 
                 num_classes: int,
                 camera_num: int,
                 view_num: int,
                 transformer_type: str,
                 semantic_weight: float = 0.2,
                 img_size: tuple = (384, 128),
                 drop_path_rate: float = 0.1,
                 drop_rate: float = 0.0,
                 attn_drop_rate: float = 0.0,
                 pretrained: Optional[str] = None,
                 convert_weights: bool = True):
        super(TransformerModel, self).__init__()
        
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.transformer_type = transformer_type
        self.semantic_weight = semantic_weight
        
        # Transformer 팩토리
        transformer_factory = {
            'swin_base_patch4_window7_224': swin_base_patch4_window7_224,
            'swin_small_patch4_window7_224': swin_small_patch4_window7_224,
            'swin_tiny_patch4_window7_224': swin_tiny_patch4_window7_224,
        }
        
        if transformer_type not in transformer_factory:
            raise ValueError(f"Unsupported transformer type: {transformer_type}")
        
        # SIE 설정
        if camera_num > 1 and view_num > 1:
            sie_camera = camera_num
            sie_view = view_num
        elif camera_num > 1:
            sie_camera = camera_num
            sie_view = 0
        elif view_num > 1:
            sie_camera = 0
            sie_view = view_num
        else:
            sie_camera = 0
            sie_view = 0
        
        # Transformer 백본 생성
        if 'swin' in transformer_type:
            self.base = transformer_factory[transformer_type](
                img_size=img_size,
                drop_path_rate=drop_path_rate,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                semantic_weight=semantic_weight
            )
        else:  # ViT
            self.base = transformer_factory[transformer_type](
                img_size=img_size,
                camera=sie_camera,
                view=sie_view,
                drop_path_rate=drop_path_rate,
                local_feature=False,
                sie_xishu=1.5
            )
        
        # 특징 차원 설정
        if hasattr(self.base, 'num_features'):
            if isinstance(self.base.num_features, list):
                self.in_planes = self.base.num_features[-1]
            else:
                self.in_planes = self.base.num_features
        else:
            self.in_planes = getattr(self.base, 'in_planes', 768)
        
        # 분류기 설정
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        
        # BNNeck 설정
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
        # 드롭아웃 설정
        self.dropout = nn.Dropout(drop_rate)
        
        # 사전 훈련된 가중치 로드
        if pretrained:
            self.load_param(pretrained)
    
    def to(self, device):
        """모델을 지정된 디바이스로 이동"""
        super().to(device)
        # 모든 하위 모듈도 같은 디바이스로 이동
        self.base.to(device)
        self.classifier.to(device)
        self.bottleneck.to(device)
        self.dropout.to(device)
        return self
    
    def forward(self, x, label=None, cam_label=None, view_label=None):
        """순전파"""
        global_feat, featmaps = self.base(x)
        
        # BNNeck 적용
        feat = self.bottleneck(global_feat)
        feat_cls = self.dropout(feat)
        
        if self.training:
            cls_score = self.classifier(feat_cls)
            return cls_score, global_feat, featmaps
        else:
            return feat, featmaps
    
    def load_param(self, trained_path: str):
        """사전 훈련된 가중치 로드"""
        param_dict = torch.load(trained_path, map_location='cpu')
        
        # state_dict 추출
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        elif 'model' in param_dict:
            param_dict = param_dict['model']
        
        # 모듈 접두사 제거
        if list(param_dict.keys())[0].startswith('module.'):
            param_dict = {k[7:]: v for k, v in param_dict.items()}
        
        # 가중치 로드
        loaded_count = 0
        for k, v in param_dict.items():
            if 'classifier' in k or 'bottleneck' in k:
                continue  # 분류기와 BNNeck은 건너뛰기
            
            try:
                if k in self.state_dict():
                    self.state_dict()[k].copy_(v)
                    loaded_count += 1
                else:
                    # 백본 모델에 직접 로드 시도
                    if hasattr(self.base, 'state_dict'):
                        base_state_dict = self.base.state_dict()
                        if k in base_state_dict:
                            base_state_dict[k].copy_(v)
                            loaded_count += 1
            except Exception as e:
                logging.warning(f"Failed to load parameter {k}: {e}")
        
        logging.info(f"Loaded {loaded_count} parameters from {trained_path}")


def make_model(cfg, 
               num_class: int, 
               camera_num: int, 
               view_num: int, 
               semantic_weight: float) -> nn.Module:
    """모델 생성 팩토리 함수"""
    
    # 설정에서 파라미터 추출
    transformer_type = cfg.MODEL.transformer_type
    img_size = tuple(cfg.INPUT.size_train)
    drop_path_rate = getattr(cfg.MODEL, 'drop_path', 0.1)
    drop_rate = getattr(cfg.MODEL, 'drop_out', 0.0)
    attn_drop_rate = getattr(cfg.MODEL, 'att_drop_rate', 0.0)
    
    # 모델 생성
    model = TransformerModel(
        num_classes=num_class,
        camera_num=camera_num,
        view_num=view_num,
        transformer_type=transformer_type,
        semantic_weight=semantic_weight,
        img_size=img_size,
        drop_path_rate=drop_path_rate,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate
    )
    
    logging.info(f"Created {transformer_type} model with {num_class} classes")
    return model
