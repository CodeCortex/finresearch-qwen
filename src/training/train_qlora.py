import os, json, math, random
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

MODEL_NAME=os.getenv('MODEL_NAME','Qwen/Qwen3-4B')
DATA_DIR=os.getenv('DATA_DIR','data/processed')
OUTPUT_DIR=os.getenv('OUTPUT_DIR','artifacts/finresearch-qwen')
SEED=int(os.getenv('SEED','42'))
MAX_LENGTH=int(os.getenv('MAX_LENGTH','2048'))
set_seed(SEED)
random.seed(SEED) 
np.random.seed(SEED)



def main():
    if not torch.cuda.is_available(): 
        raise RuntimeError('CUDA GPU is required for QLoRA training.')
    
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print('GPU:', torch.cuda.get_device_name(0), 'dtype:', dtype)
    
    
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token=tokenizer.eos_token
    
    bnb=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
    model=AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb, device_map='auto', torch_dtype=dtype)
    model.config.use_cache=False

    peft=LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM', target_modules=['q_proj','k_proj','v_proj','o_proj'])
    ds=load_dataset('json', data_files={'train':f'{DATA_DIR}/train.jsonl','validation':f'{DATA_DIR}/validation.jsonl'})
    
    args=SFTConfig(output_dir=OUTPUT_DIR, num_train_epochs=2, per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=16, learning_rate=1e-4, weight_decay=0.01, warmup_ratio=0.05, lr_scheduler_type='cosine', logging_steps=10, eval_strategy='steps', eval_steps=100, save_strategy='steps', save_steps=100, save_total_limit=2, max_length=MAX_LENGTH, completion_only_loss=True, bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(), gradient_checkpointing=True, optim='paged_adamw_8bit', report_to='none', seed=SEED)
    trainer=SFTTrainer(model=model,args=args,train_dataset=ds['train'],eval_dataset=ds['validation'],processing_class=tokenizer,peft_config=peft)
    trainer.train()
    
    
    metrics=trainer.evaluate()
    Path(OUTPUT_DIR).mkdir(parents=True,exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(Path(OUTPUT_DIR)/'eval_metrics.json','w') as f: 
        json.dump({k:float(v) if isinstance(v,(np.floating,float)) else v for k,v in metrics.items()},f,indent=2)
        
        
    if 'eval_loss' in metrics: 
        print('eval_perplexity=', math.exp(metrics['eval_loss']) if metrics['eval_loss']<20 else float('inf'))



if __name__=='__main__': 
    main()