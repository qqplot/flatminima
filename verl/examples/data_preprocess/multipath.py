import os
import argparse
from datasets import load_dataset

def make_map_fn(data_source):
    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."
    
    def process_fn(example, idx):
        base_instruction = example.get('instruction', "").strip()
        question_text = f"{base_instruction}\n\n{instruction_following}"
        
        answer_text = example.get('output', "")
        ground_truth = example.get('answer', "")
        metadata = example.get('metadata', {})
        
        if "</think>" in answer_text and not answer_text.lstrip().startswith("<think>"):
            answer_text = "<think>\n" + answer_text.lstrip()
        
        data = {
            "extra_info": {
                "question": question_text,
                "answer": answer_text,
                "ground_truth": ground_truth,
                "index": idx,
                "domain": "math",
                "difficulty": "competition", 
                "source": metadata.get('problem_id', 'unknown'),
                "model_source": metadata.get('model_source', 'unknown')
            },
            "data_source": data_source,
            "ability": "math",
        }
        return data

    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/aime_reasoning')
    parser.add_argument('--max_char_len', type=int, default=30000, help="질문 + 답변 최대 글자 수")
    args = parser.parse_args()

    data_source = 'qqplot23/aime-balanced-reasoning-10-responses'
    print(f"Loading {data_source}...", flush=True)
    
    raw_dataset = load_dataset(data_source, split='train')

    if not os.path.exists(args.local_dir):
        os.makedirs(args.local_dir)

    print("Processing AIME math dataset...")
    
    # 1. 포맷 매핑
    processed_ds = raw_dataset.map(
        function=make_map_fn(data_source), 
        with_indices=True,
        remove_columns=raw_dataset.column_names
    )
    
    original_size = len(processed_ds)
    
    # 2. 필터링 로직
    def filter_length(example):
        q_len = len(example['extra_info']['question'])
        a_len = len(example['extra_info']['answer'])
        return (q_len + a_len) <= args.max_char_len

    print(f"Filtering data exceeding {args.max_char_len:,} characters...")
    filtered_ds = processed_ds.filter(filter_length)
    filtered_size = len(filtered_ds)
    
    # 3. 필터링 후 통계 계산 및 출력
    lengths = [len(ex['extra_info']['question']) + len(ex['extra_info']['answer']) for ex in filtered_ds]
    
    # 유니크한 질문 추출 (Set 자료형 사용)
    unique_questions = set(ex['extra_info']['question'] for ex in filtered_ds)
    num_unique_questions = len(unique_questions)
    
    print("\n📊 --- 데이터셋 필터링 및 길이 통계 ---")
    print(f"원본 데이터 총 개수: {original_size:,} 개")
    print(f"제거된 데이터 개수: {original_size - filtered_size:,} 개 ({(original_size - filtered_size) / original_size * 100:.2f}%)")
    print(f"최종 남은 데이터 개수: {filtered_size:,} 개")
    print("-" * 40)
    print(f"유니크한 질문(문제) 개수: {num_unique_questions:,} 개")
    
    if num_unique_questions > 0:
        print(f"질문당 평균 답변 수: {filtered_size / num_unique_questions:.2f} 개")
    print("-" * 40)
    
    if lengths:
        max_len = max(lengths)
        min_len = min(lengths)
        avg_len = sum(lengths) / len(lengths)
        print(f"[필터링 후] 최대 길이: {max_len:,} 자")
        print(f"[필터링 후] 최소 길이: {min_len:,} 자")
        print(f"[필터링 후] 평균 길이: {avg_len:,.2f} 자")
    print("---------------------------------------\n")
    
    # 4. Parquet 파일로 저장
    output_path = os.path.join(args.local_dir, 'train_math.parquet')
    filtered_ds.to_parquet(output_path)
    print(f"✅ Saved {filtered_size:,} valid samples to {output_path}")

    print("🚀 All processing finished!")