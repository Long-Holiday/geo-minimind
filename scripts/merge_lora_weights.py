import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights with base model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        help="Base model name or path")
    parser.add_argument("--lora_model", type=str, default="out/qwen_lora_sft",
                        help="Path to trained LoRA adapter")
    parser.add_argument("--output_dir", type=str, default="out/qwen_sft_merged",
                        help="Output directory for merged model")
    args = parser.parse_args()

    # 优先检测 ModelScope 本地缓存以规避 SSL 网络限制
    base_model_path = args.base_model
    if base_model_path in ["Qwen/Qwen2.5-Coder-1.5B-Instruct", "qwen/Qwen2.5-Coder-1.5B-Instruct"]:
        modelscope_cache = os.path.expanduser("~/.cache/modelscope/hub/qwen/Qwen2.5-Coder-1.5B-Instruct")
        local_pretrained = "pretrained_models/Qwen/Qwen2.5-Coder-1.5B-Instruct"
        local_pretrained_alt = "pretrained_models/Qwen/Qwen2___5-Coder-1___5B-Instruct"
        if os.path.exists(modelscope_cache):
            print(f"Redirecting base_model {base_model_path} to local ModelScope cache: {modelscope_cache}")
            base_model_path = modelscope_cache
        elif os.path.exists(local_pretrained):
            print(f"Redirecting base_model {base_model_path} to local pre-downloaded path: {local_pretrained}")
            base_model_path = local_pretrained
        elif os.path.exists(local_pretrained_alt):
            print(f"Redirecting base_model {base_model_path} to local pre-downloaded path: {local_pretrained_alt}")
            base_model_path = local_pretrained_alt

    kwargs = {}
    if base_model_path != args.base_model:
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
