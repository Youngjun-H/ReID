from PIL import Image
from models import SOLIDEREmbeddingExtractor

def main():    
    # 더미 모델 저장
    model_path = 'reid_embedding_extractor/checkpoints/swin_base_msmt17.pth'
    config_path = 'models/solider/configs/msmt17/swin_base.yml'
    
    # 2. 임베딩 추출기 초기화 (완전히 독립적인 구현)
    print("\n1.ReID Embedding Extractor 초기화...")
    extractor = SOLIDEREmbeddingExtractor(
        model_path=model_path,
        config_path=config_path,
        device="cuda",
        semantic_weight=0.2,
        image_size=(384, 128),
        normalize_features=True    )

    # 3. 테스트 이미지 생성 (실제 이미지 파일이 없는 경우)
    print("\n2. 테스트 이미지 생성...")
    test_image = Image.new('RGB', (128, 384), color='red')
    print(f"   ✓ 테스트 이미지 생성: {test_image.size}")
    
    # 4. 임베딩 추출
    print("\n3. 임베딩 추출...")
    try:
        embedding = extractor.extract_embedding(test_image)
        print(f"   ✓ 임베딩 추출 성공!")
        print(f"   - 임베딩 차원: {len(embedding)}")
        print(f"   - 완전히 독립적인 구현으로 정상 작동")
    except Exception as e:
        print(f"   ✗ 임베딩 추출 실패: {e}")
        return

if __name__ == "__main__":
    main()