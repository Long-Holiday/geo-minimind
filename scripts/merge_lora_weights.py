import os
import sys
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 将项目根目录添加到 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from trainer.trainer_utils import resolve_model_path

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--lora_model", type=str, default="out/qwen_lora_sft",
                        help="Path to trained LoRA adapter")
    parser.add_argument("--output_dir", type=str, default="out/qwen_sft_merged",
                        help="Output directory for merged model")
    args = parser.parse_args()

    base_model_path = resolve_model_path(args.base_model)

    kwargs = {}
    if os.path.exists(base_model_path):
        kwargs["local_files_only"] = True

    print(f"Loading tokenizer from lora_model: {args.lora_model} (or base: {base_model_path})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.lora_model, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True, **kwargs)

    print(f"Loading base model from {base_model_path}...")
    device_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using dtype: {device_dtype}")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=device_dtype,
        device_map="cpu" if not torch.cuda.is_available() else "auto",
        **kwargs
    )

    print(f"Loading LoRA adapter from {args.lora_model}...")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_model,
        torch_dtype=device_dtype
    )

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Merge and save completed successfully!")

if __name__ == "__main__":
    main()
