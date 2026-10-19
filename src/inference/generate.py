from __future__ import annotations

import os
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from functools import lru_cache

BASE_MODEL=os.getenv('MODEL_NAME','Qwen/Qwen3-4B')
ADAPTER_DIR=os.getenv('ADAPTER_DIR','artifacts/finresearch-qwen')
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))

@lru_cache(maxsize=1)
def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GPU inference is required by this configuration. "
            "Run the model API on an NVIDIA CUDA machine/Colab runtime."
        )
        
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
    )
     
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=quant,
        device_map="auto",
        torch_dtype=dtype,
    )
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()
    return tokenizer, model






def _build_prompt(question: str, context: str) -> str:
    system = (
        "You are FinResearch-Qwen, a financial research assistant. "
        "Use evidence, show calculations, state uncertainty, and do not provide personalized investment advice.\n\n"
    )
    user = f"Context:\n{context or 'No retrieved evidence.'}\n\nQuestion:\n{question}"

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




def generate(question: str, context: str = "", max_new_tokens: int | None = None) -> str:
    tokenizer, model = load_model()
    prompt = _build_prompt(question, context)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

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

    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()
