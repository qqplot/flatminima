import os
import argparse
import datasets
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

def extract_solution(solution_str):
    # 기존에 사용하시던 정답 추출 로직 (수학 도메인용)
    return solution_str

def make_map_fn(split, data_source):
    def process_fn(example, idx):
        # 1. conversations에서 질문과 답변 추출
        convs = example.get('conversations', [])
        question_text = ""
        answer_text = ""
        
        for turn in convs:
            if turn['from'] == 'human':
                question_text = turn['value']
            elif turn['from'] == 'gpt':
                answer_text = turn['value']

        # 2. 기존 호출 방식을 만족하는 구조 생성
        data = {
            "extra_info": {
                "question": question_text,
                "answer": answer_text,
                "index": idx,
                "domain": example.get('domain', 'unknown'),
                "difficulty": example.get('difficulty', 'unknown'),
                "source": example.get('source', 'unknown')
            },
            "data_source": data_source,
            "ability": example.get('domain', 'math'),
        }
        return data

    return process_fn

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/open_thoughts')
    # 새롭게 추가된 인자들: 최대 토큰 길이와 샘플링 개수
    parser.add_argument('--max_length', type=int, default=8192, help='허용할 최대 토큰 수')
    parser.add_argument('--num_samples', type=int, default=3000, help='도메인 당 추출할 샘플 수')
    args = parser.parse_args()

    data_source = 'open-thoughts/OpenThoughts3-1.2M'
    print(f"Loading {data_source}...", flush=True)
    raw_dataset = load_dataset(data_source, trust_remote_code=True)['train']

    # 1. Qwen 토크나이저 로드
    print("Loading tokenizer Qwen/Qwen2.5-1.5B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    # 길이 필터링용 함수
    def length_filter(example):
        # conversation 내의 텍스트를 이어붙여 전체 길이를 계산합니다.
        text = "".join([turn['value'] for turn in example.get('conversations', [])])
        # 토큰화 (return_attention_mask=False로 속도 최적화, truncation=False로 실제 토큰 수 확인)
        tokenized = tokenizer(text, truncation=False, return_attention_mask=False)
        return len(tokenized['input_ids']) <= args.max_length

    domains = ['math', 'code', 'science']
    if not os.path.exists(args.local_dir):
        os.makedirs(args.local_dir)

    processed_datasets = []

    for domain in domains:
        print(f"\nProcessing domain: {domain}...")
        
        # 도메인 필터링
        domain_ds = raw_dataset.filter(lambda x: x['domain'] == domain)
        print(f" - Total samples: {len(domain_ds)}")
        
        if len(domain_ds) > 0:
            # 2. 토크나이저 기반 길이 필터링 (다중 프로세싱으로 속도 향상)
            print(f" - Filtering by token length (max {args.max_length})...")
            domain_ds = domain_ds.filter(length_filter, num_proc=os.cpu_count())
            print(f" - Samples after length filter: {len(domain_ds)}")

            # 3. 도메인 당 지정된 갯수(3000개) 추출
            # 편향을 막기 위해 섞은 후 추출하며, 데이터가 3000개보다 적을 경우를 대비해 min 사용
            sample_size = min(args.num_samples, len(domain_ds))
            domain_ds = domain_ds.shuffle(seed=42).select(range(sample_size))
            print(f" - Sampled {sample_size} records.")

            # 4. 기존 구조로 매핑 및 기존 컬럼 제거
            processed_ds = domain_ds.map(
                function=make_map_fn('train', data_source), 
                with_indices=True,
                remove_columns=domain_ds.column_names,
                num_proc=os.cpu_count()
            )
            
            # 병합을 위해 리스트에 보관
            processed_datasets.append(processed_ds)

    # 5. 모든 도메인을 하나의 트레인셋으로 병합
    print("\nCombining all domains into a single dataset...")
    if processed_datasets:
        # 리스트에 모인 데이터셋 병합
        combined_ds = concatenate_datasets(processed_datasets)
        
        # 모델 학습 시 특정 도메인이 몰아서 나오지 않도록 전체 데이터를 한 번 섞어줍니다
        combined_ds = combined_ds.shuffle(seed=42)
        
        # 최종 단일 파케이(Parquet) 파일로 저장
        output_path = os.path.join(args.local_dir, 'train_combined.parquet')
        combined_ds.to_parquet(output_path)
        print(f"✅ Saved unified dataset ({len(combined_ds)} samples) to {output_path}")
    else:
        print("❌ No data was processed.")

    print("🚀 All processing completed!")