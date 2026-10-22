from __future__ import annotations

import os
import torch
import sys
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from functools import lru_cache


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# PYTHON PATH
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
    
# CONFIGURATION
BASE_MODEL=os.getenv('MODEL_NAME','Qwen/Qwen3-4B')
ADAPTER_DIR=os.getenv('ADAPTER_DIR','artifacts/finresearch-qwen')
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))


# RESOLVE PROJECT PATH
def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = (
            PROJECT_ROOT / path
        )
    return path.resolve()

ADAPTER_DIR = resolve_project_path(ADAPTER_DIR)


# MODEL LOADING
@lru_cache(maxsize=1)
def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU inference is required by this configuration. "
            "Run the model on an NVIDIA CUDA machine or "
            "Google Colab GPU runtime."
        )
        
    # Validate adapter
    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found:\n"
            f"{ADAPTER_DIR}\n\n"
            f"Expected structure:\n"
            f"{PROJECT_ROOT}/artifacts/"
            f"finresearch-qwen/"
        )
    
    # Compute dtype
    bf16_supported = (torch.cuda.is_bf16_supported())
    dtype = (torch.bfloat16 if bf16_supported else torch.float16)
    
    print(
        f"Loading model: {BASE_MODEL}"
    )
    print(
        f"Adapter: {ADAPTER_DIR}"
    )
    print(
        f"Compute dtype: {dtype}"
    )
    
    # 4-bit quantization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
     
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
        
    # Base model    
    base_model = (
        AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=dtype,
        )
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model.eval()
    print("QLoRA adapter loaded successfully.")
    return tokenizer, model





# PROMPT BUILDER
def build_prompt(question: str, context: str="",) -> str:
    system = (
        "You are FinResearch-Qwen, a financial research assistant. "
        "Use evidence, show calculations, state uncertainty, and do not provide personalized investment advice.\n\n"
    )
    user = (
        f"Financial context:\n"
        f"{context or 'No retrieved evidence.'}\n\n"
        f"Question:\n"
        f"{question}"
    )
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    tokenizer, _ = load_model()
    

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )




def generate(question: str, context: str = "", max_new_tokens: int | None = None,) -> str:
    tokenizer, model = load_model()
    prompt = build_prompt(question, context)
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }
    
    
    
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
     
        
    # Remove prompt tokens 
    generated_tokens = output[0][inputs["input_ids"].shape[1]:]
    
    # Decode
    response= tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return response.strip()
