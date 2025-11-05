from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from PIL import Image
from qwen_vl_utils import process_vision_info
from pathlib import Path
from collections import Counter
import os
import argparse

# 환경 변수 설정
os.environ["HUGGINGFACE_HUB_CACHE"] = "/data/reid/reid_master/cache"

# 지원하는 이미지 확장자
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

# 전역 변수로 모델과 프로세서 저장 (재사용을 위해)
_llm = None
_processor = None


def initialize_model(model_name="Qwen/Qwen3-VL-4B-Instruct", gpu_memory_utilization=0.9, max_model_len=2048):
    """
    모델과 프로세서를 초기화합니다.
    
    Args:
        model_name: 사용할 모델 이름
        gpu_memory_utilization: GPU 메모리 사용률 (기본값: 0.9)
        max_model_len: 최대 모델 길이 (기본값: 2048)
    
    Returns:
        tuple: (llm, processor)
    """
    global _llm, _processor
    
    if _processor is None:
        _processor = AutoProcessor.from_pretrained(model_name)
    
    if _llm is None:
        _llm = LLM(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )
    
    return _llm, _processor


def prepare_inputs_for_vllm(messages, processor):
    """
    vLLM에 입력할 수 있는 형식으로 메시지를 변환합니다.
    
    Args:
        messages: 사용자 메시지 리스트
        processor: AutoProcessor 객체
    
    Returns:
        dict: vLLM 입력 형식
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, 
        image_patch_size=processor.image_processor.patch_size, 
        return_video_kwargs=True, 
        return_video_metadata=True
    )
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs
    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


def get_image_files(image_dir):
    """
    디렉토리에서 모든 이미지 파일을 찾습니다.
    
    Args:
        image_dir: 이미지 디렉토리 경로
    
    Returns:
        list: 이미지 파일 경로 리스트 (정렬됨)
    """
    image_dir = Path(image_dir)
    image_files = []
    
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))
    
    return sorted(image_files)


def process_images_in_directory(
    image_dir,
    prompt_text="글자를 읽어줘.",
    llm=None,
    processor=None,
    model_name="Qwen/Qwen3-VL-4B-Instruct",
    gpu_memory_utilization=0.9,
    max_model_len=2048,
    temperature=0.8,
    max_tokens=256,
    save_label=True,
    verbose=True
):
    """
    디렉토리 내의 모든 이미지에 대해 OCR을 수행하고 가장 많이 반복된 결과를 반환합니다.
    
    Args:
        image_dir: 이미지 디렉토리 경로
        prompt_text: OCR 프롬프트 텍스트
        llm: LLM 객체 (None이면 자동 초기화)
        processor: AutoProcessor 객체 (None이면 자동 초기화)
        model_name: 모델 이름
        gpu_memory_utilization: GPU 메모리 사용률
        max_model_len: 최대 모델 길이
        temperature: 샘플링 온도
        max_tokens: 최대 토큰 수
        save_label: label.txt 파일로 저장할지 여부
        verbose: 상세 출력 여부
    
    Returns:
        dict: {
            'final_result': 최종 결과 (가장 많이 반복된 결과),
            'all_results': 모든 결과 리스트,
            'result_stats': 결과 통계 (Counter 객체),
            'image_files': 이미지 파일 경로 리스트
        }
    """
    # 모델 초기화
    if llm is None or processor is None:
        llm, processor = initialize_model(model_name, gpu_memory_utilization, max_model_len)
    
    # 이미지 파일 찾기
    image_files = get_image_files(image_dir)
    
    if len(image_files) == 0:
        if verbose:
            print(f"경고: {image_dir}에 이미지 파일을 찾을 수 없습니다.")
        return {
            'final_result': None,
            'all_results': [],
            'result_stats': Counter(),
            'image_files': []
        }
    
    if verbose:
        print(f"총 {len(image_files)}개의 이미지 파일을 찾았습니다.\n")
    
    # 모든 이미지에 대한 입력 준비
    inputs = []
    for img_path in image_files:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(img_path)},
                    {"type": "text", "text": prompt_text}
                ]
            },
        ]
        inputs.append(prepare_inputs_for_vllm(messages, processor))
    
    # 샘플링 파라미터 설정
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    
    # 배치로 처리
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    # 결과 수집
    results = []
    if verbose:
        print("=" * 80)
    
    for i, output in enumerate(outputs):
        img_path = image_files[i]
        result_text = output.outputs[0].text.strip()
        results.append(result_text)
        if verbose:
            print(f"\n[이미지 {i+1}/{len(outputs)}]: {img_path.name}")
            print(f"결과: {result_text}")
            print("-" * 80)
    
    # 가장 많이 반복된 결과 찾기
    result_counter = Counter(results)
    if len(result_counter) > 0:
        most_common_result, most_common_count = result_counter.most_common(1)[0]
    else:
        most_common_result = None
        most_common_count = 0
    
    # 결과 통계 출력
    if verbose:
        print("\n" + "=" * 80)
        print("📊 결과 통계")
        print("=" * 80)
        if len(results) > 0:
            print(f"\n총 {len(results)}개의 결과 중 가장 많이 반복된 결과:")
            print(f"  결과: '{most_common_result}'")
            print(f"  반복 횟수: {most_common_count}회 ({most_common_count/len(results)*100:.1f}%)")
            
            print(f"\n상위 5개 결과:")
            for result, count in result_counter.most_common(5):
                print(f"  '{result}': {count}회 ({count/len(results)*100:.1f}%)")
        
        print("\n" + "=" * 80)
        print(f"✅ 최종 결과: '{most_common_result}'")
        print("=" * 80)
    
    # label.txt 저장
    if save_label and most_common_result is not None:
        label_file = Path(image_dir) / "label.txt"
        with open(label_file, 'w', encoding='utf-8') as f:
            f.write(most_common_result)
        if verbose:
            print(f"\n💾 최종 결과가 '{label_file}'에 저장되었습니다.")
    
    return {
        'final_result': most_common_result,
        'all_results': results,
        'result_stats': result_counter,
        'image_files': image_files
    }


def main():
    """CLI 인터페이스"""
    parser = argparse.ArgumentParser(description='디렉토리 내 이미지에 대해 OCR을 수행합니다.')
    parser.add_argument('image_dir', type=str, help='이미지 디렉토리 경로')
    parser.add_argument('--prompt', type=str, default='글자를 읽어줘.', help='OCR 프롬프트 텍스트')
    parser.add_argument('--model', type=str, default='Qwen/Qwen3-VL-4B-Instruct', help='모델 이름')
    parser.add_argument('--gpu-memory', type=float, default=0.9, help='GPU 메모리 사용률')
    parser.add_argument('--max-model-len', type=int, default=2048, help='최대 모델 길이')
    parser.add_argument('--temperature', type=float, default=0.8, help='샘플링 온도')
    parser.add_argument('--max-tokens', type=int, default=256, help='최대 토큰 수')
    parser.add_argument('--no-save', action='store_true', help='label.txt 저장하지 않기')
    parser.add_argument('--quiet', action='store_true', help='상세 출력 비활성화')
    
    args = parser.parse_args()
    
    result = process_images_in_directory(
        image_dir=args.image_dir,
        prompt_text=args.prompt,
        model_name=args.model,
        gpu_memory_utilization=args.gpu_memory,
        max_model_len=args.max_model_len,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        save_label=not args.no_save,
        verbose=not args.quiet
    )
    
    if args.quiet:
        print(result['final_result'])


if __name__ == "__main__":
    main()
