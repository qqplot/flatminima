import os
import datasets
import argparse
from datasets import load_dataset

def extract_solution(solution_str):
    # 기존에 사용하시던 정답 추출 로직 (수학 도메인용)
    # 여기서는 생략하지만, 필요시 상단에 정의된 함수를 그대로 쓰시면 됩니다.
    return solution_str # 혹은 정답만 추출하는 로직 적용

def make_map_fn(split, data_source):
    def process_fn(example, idx):
        # 1. conversations에서 질문과 답변 추출
        convs = example.pop('conversations')
        question_text = ""
        answer_text = ""
        
        for turn in convs:
            if turn['from'] == 'human':
                question_text = turn['value']
            elif turn['from'] == 'gpt':
                answer_text = turn['value']

        # 2. 기존 호출 방식(extra_info['question'])을 만족하는 구조 생성
        # 이 구조가 Parquet의 'extra_info' 컬럼 하나에 딕셔너리로 들어갑니다.
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
    args = parser.parse_args()

    data_source = 'open-thoughts/OpenThoughts3-1.2M'
    print(f"Loading {data_source}...", flush=True)
    raw_dataset = load_dataset(data_source, trust_remote_code=True)['train']

    domains = ['math', 'code', 'science']
    if not os.path.exists(args.local_dir):
        os.makedirs(args.local_dir)

    for domain in domains:
        print(f"Processing domain: {domain}...")
        
        # 도메인 필터링
        domain_ds = raw_dataset.filter(lambda x: x['domain'] == domain)
        
        if len(domain_ds) > 0:
            # 매핑 및 기존 컬럼 제거 (오직 'extra_info' 등 필요한 컬럼만 남김)
            processed_ds = domain_ds.map(
                function=make_map_fn('train', data_source), 
                with_indices=True,
                remove_columns=domain_ds.column_names
            )
            
            # 저장
            output_path = os.path.join(args.local_dir, f'train_{domain}.parquet')
            processed_ds.to_parquet(output_path)
            print(f"✅ Saved {len(processed_ds)} samples to {output_path}")

    print("🚀 All domains processed!")