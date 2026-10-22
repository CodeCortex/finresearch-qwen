import argparse, json, re
from pathlib import Path
from collections import Counter
from src.inference.generate import generate

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "test.jsonl"
)

DEFAULT_RESULTS_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
)

def norm(s): 
    """
    Normalize text before exact comparison.

    Removes:
    - uppercase/lowercase differences
    - special characters
    - extra spaces
    """
    return re.sub(r'[^a-z0-9.%\- ]','',s.lower()).strip()


def numeric(s):
    """
    Extract the last numeric value from text.

    Examples:
        12.5
        45%
        -10.2
    """
    vals=re.findall(r'-?\d+(?:\.\d+)?%?',s.replace(',',''))
    return vals[-1] if vals else None


def load_jsonl(path, limit):
    """
    Load evaluation records from JSONL.
    """

    if not path.exists():

        raise FileNotFoundError(
            f"Test dataset not found:\n{path}\n\n"
            f"Project root:\n{PROJECT_ROOT}"
        )

    rows = []

    with path.open(
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            if not line.strip():
                continue

            rows.append(
                json.loads(line)
            )

            if limit and len(rows) >= limit:
                break

    return rows



def main():
    ap=argparse.ArgumentParser(
        description=(
            "Evaluate FinResearch-Qwen "
            "on the financial test dataset."
        )
    ) 
    ap.add_argument(
        "--file",
        type=Path,
        default=DEFAULT_TEST_FILE,
        help=(
            "Path to test.jsonl. "
            "Relative paths are resolved from "
            "the project root."
        )
    )
    ap.add_argument('--limit',type=int,default=100, help=("Number of test samples to evaluate, 0 = all")) 
    
    args=ap.parse_args()
    rows=[json.loads(x) for x in open(arg s.file,encoding='utf-8')][:args.limit]
    exact=num_ok=0
    outputs=[]
    for i,r in enumerate(rows,1):
        pred=generate(r['prompt'], '')
        e=norm(r['completion'])
        p=norm(pred)
        exact += int(e==p)
        ev=numeric(e)
        pv=numeric(p)
        num_ok += int(ev is not None and ev==pv)
        outputs.append({'id':i,'reference':r['completion'],'prediction':pred,'exact_match':e==p,'numeric_match':ev is not None and ev==pv})
        print(i, outputs[-1]['numeric_match'])
    
    result={'n':len(rows),'exact_match':exact/len(rows) if rows else 0,'numeric_match':num_ok/len(rows) if rows else 0}
    Path('evaluation/results').mkdir(parents=True,exist_ok=True)
    json.dump({'metrics':result,'samples':outputs},open('evaluation/results/benchmark.json','w'),indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()

