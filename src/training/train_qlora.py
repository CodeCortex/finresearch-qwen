import os, json, math, random
import sys
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, set_seed)
from peft import LoraConfig
from trl import (SFTConfig, SFTTrainer)



PROJECT_ROOT = Path(__file__).resolve().parents[2]

# PYTHON PATH
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


MODEL_NAME=os.getenv('MODEL_NAME','Qwen/Qwen3-4B')
DATA_DIR=os.getenv('DATA_DIR','data/processed')
OUTPUT_DIR=os.getenv('OUTPUT_DIR','artifacts/finresearch-qwen')
SEED=int(os.getenv('SEED','42'))
MAX_LENGTH=int(os.getenv('MAX_LENGTH','2048'))



# RESOLVE PROJECT PATHS
def resolve_project_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


DATA_DIR = resolve_project_path(
    DATA_DIR
)

OUTPUT_DIR = resolve_project_path(
    OUTPUT_DIR
)


set_seed(SEED)
random.seed(SEED) 
np.random.seed(SEED)



def main():
    
    # CONFIGURATION INFORMATION
    print()
    print("=" * 70)
    print("FinResearch-Qwen QLoRA Training")
    print("=" * 70)
    print(
        f"Project root : {PROJECT_ROOT}"
    )
    print(
        f"Model        : {MODEL_NAME}"
    )
    print(
        f"Data dir     : {DATA_DIR}"
    )
    print(
        f"Output dir   : {OUTPUT_DIR}"
    )
    print(
        f"Seed         : {SEED}"
    )
    print(
        f"Max length   : {MAX_LENGTH}"
    )
    print("=" * 70)
    print()
    
    
    if not torch.cuda.is_available(): 
        raise RuntimeError(
            "CUDA GPU is required for QLoRA training. "
            "Use an NVIDIA GPU environment such as "
            "Google Colab or Kaggle."
        )
    
    # DEVICE / DTYPE
    bf16_supported = (
        torch.cuda.is_bf16_supported()
    )
    dtype=(torch.bfloat16 if bf16_supported else torch.float16)

    
    print('GPU:', torch.cuda.get_device_name(0))
    print("Compute dtype:", dtype)
    print()
    
    # VALIDATE DATASET PATHS
    train_file= (DATA_DIR/"train.jsonl")
    validation_file= (DATA_DIR/"validation.jsonl")
    
    if not train_file.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n"
            f"{train_file}\n\n"
            f"Expected project structure:\n"
            f"{PROJECT_ROOT}/data/processed/train.jsonl"
        )
        
    if not validation_file.exists():
        raise FileNotFoundError(
            f"Validation dataset not found:\n"
            f"{validation_file}\n\n"
            f"Expected project structure:\n"
            f"{PROJECT_ROOT}/data/processed/validation.jsonl"
        )
    
    print("Training dataset:", train_file)
    print("Valdation dataset:", validation_file)
    print()
    
    
    #TOKENIZER
    print("Loading tokenizer....")
    tokenizer=AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token=tokenizer.eos_token
        
    
    # 4-BIT QUANTIZATION
    print("Configuring 4-bit NF4 quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
    
    # LOAD BASE MODEL
    print("Loading base model...")
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=dtype,
        )
    )
    model.config.use_cache = False
    
    # LORA CONFIGURATION
    print("Configuring LoRA...")
    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )
    
    # LOAD DATASET
    print("Loading training dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(
                train_file
            ),
            "validation": str(
                validation_file
            ),
        },
    )

    print(f"Train samples      : "f"{len(dataset['train'])}")
    print(f"Validation samples : "f"{len(dataset['validation'])}")
    print()
    
    
    # TRAINING CONFIGURATION
    training_args = SFTConfig(
        output_dir=str(
            OUTPUT_DIR
        ),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-4,
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        max_length=MAX_LENGTH,
        completion_only_loss=True,
        bf16=bf16_supported,
        fp16=not bf16_supported,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="none",
        seed=SEED,
    )
    
    # CREATE TRAINER
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    # TRAIN
    print()
    print("=" * 70)
    print("Starting QLoRA training...")
    print("=" * 70)
    print()

    trainer.train()
    
    
    # EVALUATION
    print()
    print("Running validation evaluation...")
    metrics=trainer.evaluate()
    
    
    OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
    
    # SAVE LORA ADAPTER
    print("Saving LoRA adapter...")
    trainer.save_model(str(OUTPUT_DIR))
    
    
    #SAVE TOKENIZER
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    
    
    # SAVE EVALUATION METRICS
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.floating):
            serializable_metrics[key] = (
                float(value)
            )
        elif isinstance(value, (float, int)):
            serializable_metrics[key] = value
        else:
            serializable_metrics[key] = value
    
    metrics_file = (OUTPUT_DIR/ "eval_metrics.json")
    with metrics_file.open("w", encoding="utf-8") as file:
        json.dump(serializable_metrics, file, indent=2)
        
    
    # PERPLEXITY
    if "eval_loss" in metrics:
        eval_loss = metrics["eval_loss"]

        if eval_loss < 20:
            perplexity = math.exp(eval_loss)
        else:
            perplexity = float("inf")

        print(f"eval_perplexity = "f"{perplexity:.4f}")
    
    
    # FINAL OUTPUT
    print()
    print("=" * 70)
    print("QLoRA training complete")
    print("=" * 70)
    print(
        f"Model adapter : {OUTPUT_DIR}"
    )
    print(
        f"Metrics       : {metrics_file}"
    )
    print("=" * 70)
        
    
if __name__=='__main__': 
    main()