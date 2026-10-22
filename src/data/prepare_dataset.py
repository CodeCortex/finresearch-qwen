import argparse, json, random
from pathlib import Path
from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

SYSTEM = "You are FinResearch-Qwen, a financial research assistant. Give evidence-based answers, show calculations when relevant, distinguish facts from assumptions, and do not provide personalized investment advice."


def make_record(row):
    context = str(row.get('context', '')).strip()
    question = str(row.get('question', '')).strip()
    answer = str(row.get('answer', row.get('output', ''))).strip()
    if not question or not answer:
        return None
    prompt = f"{SYSTEM}\n\nFinancial context:\n{context}\n\nQuestion:\n{question}"
    return {"prompt": prompt, "completion": answer}

def load_source(name, split):
    ds = load_dataset(name, split=split)
    return [make_record(x) for x in ds]


def dedupe(records):
    seen=set(); out=[]
    for r in records:
        if not r: 
            continue
        key=(r['prompt'].strip().lower(), r['completion'].strip().lower())
        if key not in seen:
            seen.add(key) 
            out.append(r)
    return out

def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows: 
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def main():
    ap=argparse.ArgumentParser(
        description=(
            "Prepare financial QA dataset "
            "for FinResearch-Qwen QLoRA training."
        )
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Output directory. Relative paths are "
            "resolved from the project root."
        )
    )
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=0, help='0 = all')
    args=ap.parse_args()
    random.seed(args.seed)
    
    output_dir = args.output_dir

    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    output_dir = output_dir.resolve()
    
    print()
    print("=" * 60)
    print("FinResearch-Qwen Dataset Preparation")
    print("=" * 60)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Output dir   : {output_dir}")
    print(f"Seed         : {args.seed}")
    print(f"Limit        : {args.limit or 'ALL'}")
    print("=" * 60)
    print()
    
    
    dataset_name = (
        "hellotayssir/"
        "FinQA_TAT-QA_financial_finetuning_dataset"
    )

    train = load_source(
        dataset_name,
        "train"
    )

    test = load_source(
        dataset_name,
        "test"
    )
    # train = load_source('hellotayssir/FinQA_TAT-QA_financial_finetuning_dataset', 'train')
    # test = load_source('hellotayssir/FinQA_TAT-QA_financial_finetuning_dataset', 'test')
    
    
    
    train=dedupe(train)
    test=dedupe(test)
    
    print(
        f"After deduplication:"
    )

    print(
        f"Train source : {len(train)}"
    )

    print(
        f"Test source  : {len(test)}"
    )
    
    random.shuffle(train)
    random.shuffle(test)
    if args.limit: train=train[:args.limit]
    
    
    n_val=max(1, int(len(train)*0.10))
    val=train[:n_val]
    train=train[n_val:]
    # out=Path(args.output_dir)
    # out = output_dir
    # write_jsonl(out/'train.jsonl', train)
    # write_jsonl(out/'validation.jsonl', val)
    # write_jsonl(out/'test.jsonl', test)
    # print(f'train={len(train)} validation={len(val)} test={len(test)}')
    
    train_path = (
        output_dir / "train.jsonl"
    )

    validation_path = (
        output_dir / "validation.jsonl"
    )

    test_path = (
        output_dir / "test.jsonl"
    )
    
    write_jsonl(
        train_path,
        train
    )

    write_jsonl(
        validation_path,
        val
    )

    write_jsonl(
        test_path,
        test
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("Dataset preparation complete")
    print("=" * 60)

    print(
        f"Train      : {len(train)}"
    )

    print(
        f"Validation : {len(val)}"
    )

    print(
        f"Test       : {len(test)}"
    )

    print()
    print(
        f"Train file      : {train_path}"
    )

    print(
        f"Validation file : {validation_path}"
    )

    print(
        f"Test file       : {test_path}"
    )

    print("=" * 60)




if __name__== '__main__':
    main()