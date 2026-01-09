"""
Merge QLoRA LoRA adapters into the base model
This creates a standalone model that doesn't require the base model + adapters
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import shutil

# Configuration
BASE_MODEL = "./gemma-3-4b-it"
LORA_PATH = "./qlora_output"
OUTPUT_PATH = "./gemma-3-4b-it-merged"


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Check GPU availability
    if torch.cuda.is_available():
        device = "cuda"
        if torch.cuda.is_bf16_supported():
            model_dtype = torch.bfloat16
            print("Using bfloat16 (GPU supports bf16)")
        else:
            model_dtype = torch.float16
            print("Using float16 (GPU doesn't support bf16)")
    else:
        device = "cpu"
        model_dtype = torch.float32
        print("No GPU detected, using CPU with float32")

    print(f"\nLoading base model from {BASE_MODEL}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=model_dtype,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Loading LoRA adapters from {LORA_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)

    print("Merging LoRA adapters into base model...")
    # Merge and unload (this merges the adapters and removes the LoRA layers)
    merged_model = model.merge_and_unload()

    print(f"\nSaving merged model to {OUTPUT_PATH}...")
    # Save the merged model
    merged_model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_PATH)

    # Copy all necessary config files if they exist
    config_files = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "preprocessor_config.json",  # Required for multimodal models
        "processor_config.json",     # Required for multimodal models
        "chat_template.json",
        "special_tokens_map.json",
        "added_tokens.json"
    ]
    for config_file in config_files:
        src = os.path.join(BASE_MODEL, config_file)
        dst = os.path.join(OUTPUT_PATH, config_file)
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)
            print(f"  Copied {config_file}")

    print(f"\n✓ Merged model saved to {OUTPUT_PATH}")
    print("You can now use this model without needing the base model + LoRA adapters separately.")
    print(
        f"\nModel size: {sum(p.numel() * p.element_size() for p in merged_model.parameters()) / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
