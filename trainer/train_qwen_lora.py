import os
import sys
import glob
import torch
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

# 将项目根目录添加到 sys.path 中
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from dataset.gee_dataset import GEESFTDataset

# 设置环境变量以防 tokenizer 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_latest_checkpoint(output_dir):
    """
    扫描输出目录，自动定位到最新保存的 checkpoint 子目录
    """
    if not os.path.exists(output_dir):
        return None
    checkpoints = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoints:
        return None
    # 提取数字步数并排序，返回步数最大的 checkpoint 路径
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return checkpoints[-1]

def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-Coder-1.5B-Instruct LoRA SFT Trainer")
    parser.add_argument("--model_name_or_path", "--model_path", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct",
                        dest="model_name_or_path", help="HuggingFace model path or local path")
    parser.add_argument("--train_file", "--data_path", type=str, default="data/gee_sft_merged_train.jsonl",
                        dest="train_file", help="Path to training jsonl file")
    parser.add_argument("--val_file", "--eval_data_path", type=str, default="data/gee_sft_merged_val.jsonl",
                        dest="val_file", help="Path to validation jsonl file")
    parser.add_argument("--output_dir", type=str, default="out/qwen_lora_sft",
                        help="Output directory for checkpoints")
    parser.add_argument("--from_resume", action="store_true",
                        help="Resume training from the latest checkpoint if available")
    parser.add_argument("--num_train_epochs", "--epochs", type=int, default=3,
                        dest="num_train_epochs", help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", "--batch_size", type=int, default=4,
                        dest="per_device_train_batch_size", help="Batch size per device for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4,
                        help="Batch size per device for evaluation")
    parser.add_argument("--gradient_accumulation_steps", "--accumulation_steps", type=int, default=4,
                        dest="gradient_accumulation_steps", help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--dataloader_num_workers", type=int, default=4, help="Number of subprocesses for data loading")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Initial learning rate")
    parser.add_argument("--max_length", "--max_seq_len", type=int, default=1024,
                        dest="max_length", help="Maximum sequence length")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate model every X steps")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Maximum training steps (-1 to disable)")
    parser.add_argument("--lora_r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=128, help="LoRA alpha scaling factor")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Wandb tracking")
    parser.add_argument("--wandb_project", type=str, default="geo-minimind-sft", help="Wandb project name")
    parser.add_argument("--use_swanlab", action="store_true", help="Enable Swanlab tracking")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml file")
    
    args = parser.parse_args()

    # 从 config.yaml 中读取参数覆盖默认参数
    config_file = args.config
    if not os.path.exists(config_file) and config_file == "config.yaml" and os.path.exists("config.ymal"):
        config_file = "config.ymal"

    if os.path.exists(config_file):
        try:
            import yaml
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            if config_data and 'sft' in config_data:
                print(f"Loading configuration from {config_file} for SFT training...")
                sft_config = config_data['sft']
                for key, val in sft_config.items():
                    if hasattr(args, key):
                        setattr(args, key, val)
                        print(f"  [Config SFT] Override: {key} = {val}")
        except Exception as e:
            print(f"[Warning] Failed to load config file {config_file}: {e}")

    from trainer.trainer_utils import resolve_model_path
    model_path = resolve_model_path(args.model_name_or_path)

    # 1. 实验追踪初始化
    report_to = []
    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, config=vars(args))
            report_to.append("wandb")
        except ImportError:
            print("[Warning] wandb is not installed, skipping wandb logging.")
            
    if args.use_swanlab:
        try:
            import swanlab
            swanlab.init(project=args.wandb_project, config=vars(args))
            report_to.append("swanlab")
        except ImportError:
            print("[Warning] swanlab is not installed, skipping swanlab logging.")
            
    if not report_to:
        report_to = "none"
    elif len(report_to) == 1:
        report_to = report_to[0]

    # 2. 检查并加载 Tokenizer 与 Model
    kwargs = {}
    if os.path.exists(model_path):
        kwargs["local_files_only"] = True

    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading base model from {model_path}...")
    device_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Using device precision: {device_dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=device_dtype,
        device_map="auto",
        **kwargs
    )

    # 3. 启用梯度检查点以降低显存
    model.gradient_checkpointing_enable()

    # 4. 配置 PEFT LoRA
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 5. 准备 Datasets
    print("Loading datasets...")
    train_dataset = GEESFTDataset(args.train_file, tokenizer, max_length=args.max_length)
    val_dataset = None
    if os.path.exists(args.val_file):
        val_dataset = GEESFTDataset(args.val_file, tokenizer, max_length=args.max_length)

    # 6. 配置 TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False if args.from_resume else True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=(device_dtype == torch.float16),
        bf16=(device_dtype == torch.bfloat16),
        gradient_checkpointing=True,
        eval_strategy="steps" if val_dataset is not None else "no",
        eval_steps=args.eval_steps if val_dataset is not None else None,
        dataloader_num_workers=getattr(args, 'dataloader_num_workers', 4),
        dataloader_pin_memory=True,
        report_to=report_to,
        logging_first_step=True,
        remove_unused_columns=False
    )

    callbacks = []
    if "swanlab" in report_to or args.use_swanlab:
        try:
            from swanlab.integration.huggingface import SwanLabCallback
            callbacks.append(SwanLabCallback())
        except ImportError:
            pass

    # 7. 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=callbacks
    )

    # 8. 处理断点续训
    resume_checkpoint = None
    if args.from_resume:
        latest_ckpt = get_latest_checkpoint(args.output_dir)
        if latest_ckpt:
            print(f"Found latest checkpoint: {latest_ckpt}. Resuming training...")
            resume_checkpoint = latest_ckpt
        else:
            print(f"No checkpoint found in {args.output_dir}. Starting training from scratch...")

    # 9. 开始训练
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    # 10. 保存最终模型的 PEFT adapter
    print(f"Saving final adapter to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
