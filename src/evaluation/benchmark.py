import argparse, json, re
from pathlib import Path
import sys
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# PYTHON PATH
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.inference.generate import generate



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
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help=(
            "Directory where benchmark results "
            "are saved."
        )
    )
    args=ap.parse_args()
    
    test_file = args.file
    if not test_file.is_absolute():
    
        test_file = (
            PROJECT_ROOT / test_file
        )

    test_file = test_file.resolve()
    
    results_dir = args.results_dir
    if not results_dir.is_absolute():

        results_dir = (
            PROJECT_ROOT / results_dir
        )

    results_dir = results_dir.resolve()
    
    # ========================================================
    # PRINT CONFIGURATION
    # ========================================================

    print()
    print("=" * 65)
    print("FinResearch-Qwen Evaluation")
    print("=" * 65)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Test file    : {test_file}"
    )

    print(
        f"Results dir  : {results_dir}"
    )

    print(
        f"Limit        : "
        f"{args.limit if args.limit else 'ALL'}"
    )

    print("=" * 65)
    print()

   # ========================================================
    # LOAD DATASET
    # ========================================================

    rows = load_jsonl(
        test_file,
        args.limit
    )

    if not rows:

        raise RuntimeError(
            "No evaluation records found."
        )

    print(
        f"Loaded {len(rows)} evaluation records."
    )

    print()

    

    exact_matches=numeric_matches=0
    outputs=[]
    for i,r in enumerate(rows,start=1):
        print(
            f"[{i}/{len(r)}] "
            f"Generating prediction..."
        )
        
        prediction=generate(r['prompt'], '')
        
        expected=norm(r['completion'])
        predicted=norm(prediction)
        
        is_exact_match = (
            expected == predicted
        )

        exact_matches += int(
            is_exact_match
        )
        
        # numeric match
    
        expected_numeric = numeric(
            expected
        )

        predicted_numeric = numeric(
            predicted
        )

        is_numeric_match = (
            expected_numeric is not None
            and expected_numeric == predicted_numeric
        )

        numeric_matches += int(
            is_numeric_match
        )

        outputs.append({'id':i,'reference':r['completion'],'prediction':prediction,'exact_match':is_exact_match,'numeric_match':is_numeric_match, "reference_numeric":(expected_numeric), "prediction_numeric":(predicted_numeric) })
        print(
            f"    exact_match  : "
            f"{is_exact_match}"
        )

        print(
            f"    numeric_match: "
            f"{is_numeric_match}"
        )
        
    
    total = len(rows)
    result = {

        "n": total,

        "exact_match": (
            exact_matches / total
        ),

        "numeric_match": (
            numeric_matches / total
        )
    }
    
    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_dir.mkdir(
        parents=True,
        exist_ok=True
    )
    
    result_file = (
        results_dir
        / "benchmark.json"
    )

    output = {

        "metrics": result,

        "config": {

            "test_file": str(
                test_file
            ),

            "limit": args.limit,

            "seed": 42
        },

        "samples": outputs
    }
    
    with result_file.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=2, ensure_ascii=False)
    
    
    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print("=" * 65)
    print("Evaluation Complete")
    print("=" * 65)

    print(
        f"Samples        : {total}"
    )

    print(
        f"Exact Match    : "
        f"{result['exact_match']:.4f}"
    )

    print(
        f"Numeric Match  : "
        f"{result['numeric_match']:.4f}"
    )

    print()
    print(
        f"Results saved to:"
    )

    print(
        result_file
    )

    print("=" * 65)

    

if __name__=='__main__':
    main()

