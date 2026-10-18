import argparse, json, re
from pathlib import Path
from collections import Counter
from src.inference.generate import generate


def norm(s): return re.sub(r'[^a-z0-9.%\- ]','',s.lower()).strip()
def numeric(s):
    vals=re.findall(r'-?\d+(?:\.\d+)?%?',s.replace(',',''))
    return vals[-1] if vals else None


def main():
    ap=argparse.ArgumentParser() 
    ap.add_argument('--file',default='data/processed/test.jsonl') 
    ap.add_argument('--limit',type=int,default=100) 
    args=ap.parse_args()
    rows=[json.loads(x) for x in open(args.file,encoding='utf-8')][:args.limit]
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

