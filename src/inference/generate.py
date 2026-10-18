import os
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from functools import lru_cache

BASE_MODEL=os.getenv('MODEL_NAME','Qwen/Qwen3-4B')
ADAPTER=os.getenv('ADAPTER_PATH','artifacts/finresearch-qwen')


@lru_cache(maxsize=1)
def load():
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tok=AutoTokenizer.from_pretrained(ADAPTER)
    if tok.pad_token is None: 
        tok.pad_token=tok.eos_token
        
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=dtype,bnb_4bit_use_double_quant=True)
    base=AutoModelForCausalLM.from_pretrained(BASE_MODEL,quantization_config=bnb,device_map='auto',torch_dtype=dtype)
    model=PeftModel.from_pretrained(base, ADAPTER)
    model.eval()
    
    return tok,model


def generate(question, context=''):
    tok,model=load()
    prompt=("You are FinResearch-Qwen, a financial research assistant. "
            "Use evidence, show calculations, state uncertainty, and do not provide personalized investment advice.\n\n"
            f"Context:\n{context}\n\nQuestion:\n{question}")
    
    inputs=tok(prompt,return_tensors='pt').to(model.device)
    with torch.inference_mode():
        out=model.generate(**inputs,max_new_tokens=512,do_sample=False,pad_token_id=tok.eos_token_id)
        
        
    return tok.decode(out[0][inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()


