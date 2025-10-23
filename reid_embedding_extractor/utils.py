#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
독립적인 유틸리티 함수들 - SOLIDER_REID 의존성 제거
"""

import logging
import os
import sys
from typing import Optional


def setup_logger(name: str, 
                 output_dir: Optional[str] = None, 
                 if_train: bool = True) -> logging.Logger:
    """로거 설정"""
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # 기존 핸들러 제거
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 포맷터 설정
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 콘솔 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 파일 핸들러 (출력 디렉토리가 있는 경우)
    if output_dir and os.path.exists(output_dir):
        log_file = os.path.join(output_dir, 'log.txt')
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def create_output_dir(output_dir: str) -> None:
    """출력 디렉토리 생성"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)