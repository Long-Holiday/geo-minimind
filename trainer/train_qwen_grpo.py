import sys
import os

# 设置环境变量以防 tokenizer 警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 动态将项目根目录加入到 sys.path 中以防止导入失败
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
import argparse
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 导入我们刚刚编写的 reward 模块
from trainer.gee_reward import (
    syntax_reward_func,
    api_reward_func,
    format_reward_func,
    completeness_reward_func,
    gee_reward_func
)

# 尝试导入 wandb / swanlab
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import swanlab
    SWANLAB_AVAILABLE = True
except ImportError:
    SWANLAB_AVAILABLE = False


def create_mock_data():
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    rl_prompts_path = os.path.join(data_dir, "gee_rl_prompts_train.jsonl")
    
    if not os.path.exists(rl_prompts_path):
        mock_prompts = [
            {"prompt": "Write a GEE script to compute NDVI for Beijing in 2023 using Landsat 8 images and reduce it to mean value."},
            {"prompt": "Compute the land surface temperature (LST) for Shanghai using Sentinel-3 data and clip it with the city boundary."},
            {"prompt": "Filter Sentinel-2 image collection by date '2023-06-01' to '2023-09-30', select bands and calculate the median composite."},
            {"prompt": "Extract precipitation data from CHIRPS collection for a given polygon in Tibet and export the result as table."},
            {"prompt": "Create a random forest classification map for forest coverage in Yunnan province using Landsat composites."},
            {"prompt": "Calculate the annual mean temperature using WorldClim dataset and reduce region using a rectangular bounding box."},
            {"prompt": "Filter Landsat 9 images with less than 10% cloud cover and compute EVI index, then visualize it."},
            {"prompt": "Get the DEM elevation and calculate slope and aspect for the Mount Everest region."},
            {"prompt": "Perform water body classification using Sentinel-1 SAR data and compute the water area buffer."},
            {"prompt": "Load MODIS snow cover dataset, filter by winter season and compute the snow cover frequency map."}
        ]
        with open(rl_prompts_path, "w", encoding="utf-8") as f:
            for item in mock_prompts:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Created mock prompt dataset at {rl_prompts_path}")

try:
    from dataset.gee_dataset import GEERLDataset
except ImportError:
    class GEERLDataset(Dataset):
        def __init__(self, file_path, tokenizer, max_length=1024):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.prompts = []
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            prompt = data.get("prompt", data.get("user_goal", ""))
                            if prompt:
                                self.prompts.append(prompt)
                        except Exception:
                            pass
            if not self.prompts:
                self.prompts = ["Write a GEE script to compute NDVI for Beijing in 2023."]
                
        def __len__(self):
            return len(self.prompts)
            
        def __getitem__(self, idx):
            user_goal = self.prompts[idx]
            conversations = [
                {"role": "system", "content": "你是 GEE-Coder，一个专注于 Google Earth Engine Python API 的代码生成助手。"},
                {"role": "user", "content": user_goal}
            ]
            prompt = self.tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True)
            inputs = self.tokenizer(prompt, max_length=self.max_length, truncation=True)
            return {
                'prompt': prompt,
                'input_ids': inputs.input_ids,
                'attention_mask': inputs.attention_mask
            }


def get_latest_checkpoint(output_dir):
    if not os.path.exists(output_dir):
        return None, None
    dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not dirs:
        return None, None
    steps = []
    for d in dirs:
        try:
            steps.append(int(d.split("-")[1]))
        except ValueError:
            pass
    if not steps:
        return None, None
    latest_step = max(steps)
    return os.path.join(output_dir, f"checkpoint-{latest_step}"), latest_step


def save_checkpoint(model, optimizer, step, output_dir):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.bin"))
    meta = {"step": step}
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"[Checkpoint] Saved to {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description="GRPO Alignment for Qwen2.5-Coder GEE")
    parser.add_argument("--model_name_or_path", "--model_path", type=str, default="pretrained_models/Qwen/Qwen2___5-Coder-1___5B-Instruct",
                        dest="model_name_or_path", help="Base model name or path")
    parser.add_argument("--data_path", type=str, default="./data/gee_rl_prompts_train.jsonl",
                        help="Path to RL training prompts jsonl file")
    parser.add_argument("--from_resume", action="store_true", help="Resume training from latest checkpoint")
    parser.add_argument("--output_dir", type=str, default="./out/qwen_grpo", help="Output directory")
    parser.add_argument("--num_epochs", "--epochs", type=int, default=3, dest="num_epochs", help="Number of training epochs")
    parser.add_argument("--lr", "--learning_rate", type=float, default=5e-6, dest="lr", help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (currently kept as 1 to avoid sequence padding in dataloader)")
    parser.add_argument("--beta", type=float, default=0.01, help="KL penalty coefficient")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Clipping parameter for GRPO")
    parser.add_argument("--ppo_epochs", type=int, default=1, help="Number of policy updates per batch")
    parser.add_argument("--num_generations", type=int, default=4, help="Number of generations per prompt")
    parser.add_argument("--save_steps", "--save_interval", type=int, default=10, dest="save_steps", help="Steps between checkpoints")
    parser.add_argument("--log_steps", type=int, default=1, help="Steps between logs")
    parser.add_argument("--max_seq_len", type=int, default=1024, help="Maximum prompt sequence length")
    parser.add_argument("--max_gen_len", type=int, default=512, help="Maximum generated sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Wandb tracking")
    parser.add_argument("--wandb_project", type=str, default="geo-minimind-grpo", help="Wandb project name")
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
            if config_data and 'grpo' in config_data:
                print(f"Loading configuration from {config_file} for GRPO training...")
                grpo_config = config_data['grpo']
                for key, val in grpo_config.items():
                    if hasattr(args, key):
                        setattr(args, key, val)
                        print(f"  [Config GRPO] Override: {key} = {val}")
        except Exception as e:
            print(f"[Warning] Failed to load config file {config_file}: {e}")

    device = args.device
    print(f"Using device: {device}")

    # 1. 确保 Mock 数据及路径存在
    create_mock_data()
    rl_prompts_path = args.data_path

    # 2. 确定模型路径
    from trainer.trainer_utils import resolve_model_path
    model_path = "out/qwen_sft_merged"
    if not os.path.exists(model_path) or not os.listdir(model_path):
        print(f"[Warning] SFT merged model not found at {model_path}. Fallback to base model...")
        model_path = resolve_model_path(args.model_name_or_path)

    kwargs = {}
    if os.path.exists(model_path):
        kwargs["local_files_only"] = True

    # 3. 加载分词器和模型
    print(f"Loading tokenizer and model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
        trust_remote_code=True,
        use_safetensors=True,
        **kwargs
    ).to(device)

    # 4. LoRA 适配器配置
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 5. 断点续训恢复逻辑
    latest_step = None
    lora_ckpt_path = None
    if args.from_resume:
        lora_ckpt_path, latest_step = get_latest_checkpoint(args.output_dir)

    if lora_ckpt_path is not None:
        print(f"Resuming training from checkpoint: {lora_ckpt_path} (step {latest_step})")
        model = PeftModel.from_pretrained(base_model, lora_ckpt_path, is_trainable=True)
    else:
        print("Starting fresh training with new LoRA adapter...")
        model = get_peft_model(base_model, peft_config)

    model.print_trainable_parameters()

    # 6. 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if latest_step is not None:
        opt_path = os.path.join(lora_ckpt_path, "optimizer.bin")
        if os.path.exists(opt_path):
            try:
                optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
                print("Optimizer state loaded successfully.")
            except Exception as e:
                print(f"Warning: Failed to load optimizer state: {e}")

    # 7. 加载数据集 (传给构造函数 tokenizer 和 max_length 参数)
    dataset = GEERLDataset(rl_prompts_path, tokenizer=tokenizer, max_length=args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, pin_memory=True, num_workers=2 if torch.cuda.is_available() else 0)

    # 8. 初始化实验追踪器
    tracker_initialized = False
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(project=args.wandb_project, config=vars(args))
        tracker_initialized = True
        print("Wandb initialized.")
    elif args.use_swanlab and SWANLAB_AVAILABLE:
        swanlab.init(project=args.wandb_project, config=vars(args))
        tracker_initialized = True
        print("SwanLab initialized.")

    start_step = latest_step + 1 if latest_step is not None else 1
    global_step = start_step

    print("Starting GRPO training loop...")
    for epoch in range(args.num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.num_epochs} ---")
        for batch in dataloader:
            prompt_text = batch["prompt"][0]
            
            # 使用 tokenizer 对已应用模板的 prompt 文本重新 tokenize 以获取长度
            prompt_inputs = tokenizer(prompt_text, return_tensors="pt")
            prompt_ids = prompt_inputs["input_ids"].to(device)
            prompt_len = prompt_ids.shape[1]

            # B. 采样生成 G=4 个 completion 样本
            prompt_ids_batch = prompt_ids.repeat(args.num_generations, 1).to(device)
            model.eval()
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=prompt_ids_batch,
                    max_new_tokens=args.max_gen_len,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # 截取 completion 部分，并进行 decode
            completion_ids = outputs[:, prompt_len:]
            completions = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)

            # C. 评分与计算优势 (Group Relative Advantage)
            prompts_dup = [prompt_text] * args.num_generations
            
            # 各维度打分，用于日志记录
            syntax_rewards = syntax_reward_func(prompts_dup, completions)
            api_rewards = api_reward_func(prompts_dup, completions)
            format_rewards = format_reward_func(prompts_dup, completions)
            completeness_rewards = completeness_reward_func(prompts_dup, completions)
            
            # 综合 Reward
            rewards = gee_reward_func(prompts_dup, completions)
            rewards = np.array(rewards, dtype=np.float32)
            
            mean_r = rewards.mean()
            std_r = rewards.std()
            if std_r < 1e-6:
                advantages = np.zeros_like(rewards)
            else:
                advantages = (rewards - mean_r) / (std_r + 1e-8)
            
            advantages = torch.tensor(advantages, device=device)

            # D. 建立 completion 的 mask
            attention_mask = (outputs != tokenizer.pad_token_id).long()
            completion_mask = torch.zeros_like(outputs)
            for i in range(args.num_generations):
                non_pad_indices = torch.where(attention_mask[i] == 1)[0]
                if len(non_pad_indices) > 0:
                    last_non_pad = non_pad_indices[-1].item()
                    if last_non_pad >= prompt_len:
                        completion_mask[i, prompt_len : last_non_pad + 1] = 1.0

            # E. 第一次前向计算 old_log_probs (为计算比值 ratio 做准备)
            model.eval()
            with torch.no_grad():
                old_logits = model(input_ids=outputs, attention_mask=attention_mask).logits
                shift_old_logits = old_logits[..., :-1, :].contiguous()
                shift_labels = outputs[..., 1:].contiguous()
                shift_mask = completion_mask[..., 1:].contiguous()
                
                old_log_probs = torch.log_softmax(shift_old_logits, dim=-1)
                per_token_old_log_probs = torch.gather(old_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

                # Reference 模型的概率计算（LoRA 冻结底座）
                with model.disable_adapter():
                    ref_logits = model(input_ids=outputs, attention_mask=attention_mask).logits
                    shift_ref_logits = ref_logits[..., :-1, :].contiguous()
                    ref_log_probs = torch.log_softmax(shift_ref_logits, dim=-1)
                    per_token_ref_log_probs = torch.gather(ref_log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

            # F. 多步 Policy 更新 (PPO / GRPO clipping)
            model.train()
            step_losses = []
            step_kls = []
            
            for ppo_epoch in range(args.ppo_epochs):
                logits = model(input_ids=outputs, attention_mask=attention_mask).logits
                shift_logits = logits[..., :-1, :].contiguous()
                
                log_probs = torch.log_softmax(shift_logits, dim=-1)
                per_token_log_probs = torch.gather(log_probs, dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)

                # 比值 r_t
                ratio = torch.exp(per_token_log_probs - per_token_old_log_probs)
                
                # Clipped Surrogates
                surr1 = ratio * advantages.unsqueeze(-1)
                surr2 = torch.clamp(ratio, 1.0 - args.epsilon, 1.0 + args.epsilon) * advantages.unsqueeze(-1)
                
                # KL 散度约束 (采用 Schulman 估计式以降低方差并确保非负)
                log_ratio = per_token_ref_log_probs - per_token_log_probs
                kl = torch.exp(log_ratio) - 1.0 - log_ratio
                
                # Loss 计算
                loss_t = -torch.min(surr1, surr2) + args.beta * kl
                loss = (loss_t * shift_mask).sum() / (shift_mask.sum() + 1e-8)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                step_losses.append(loss.item())
                step_kls.append(kl.detach().mean().item())

            # G. 日志记录
            if global_step % args.log_steps == 0:
                print(f"[Step {global_step}] Loss: {np.mean(step_losses):.4f} | R_mean: {mean_r:.4f} | R_std: {std_r:.4f} | KL: {np.mean(step_kls):.4f}")
                print(f"        -> Sub Rewards | Syntax: {np.mean(syntax_rewards):.2f} | API: {np.mean(api_rewards):.2f} | Format: {np.mean(format_rewards):.2f} | Completeness: {np.mean(completeness_rewards):.2f}")
                
                log_data = {
                    "step": global_step,
                    "loss": np.mean(step_losses),
                    "kl": np.mean(step_kls),
                    "reward_mean": mean_r,
                    "reward_std": std_r,
                    "reward_syntax": np.mean(syntax_rewards),
                    "reward_api": np.mean(api_rewards),
                    "reward_format": np.mean(format_rewards),
                    "reward_completeness": np.mean(completeness_rewards),
                }
                
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log(log_data)
                elif args.use_swanlab and SWANLAB_AVAILABLE:
                    swanlab.log(log_data)

            # H. 保存 Checkpoint
            if global_step % args.save_steps == 0:
                save_checkpoint(model, optimizer, global_step, args.output_dir)

            global_step += 1

    # 保存最终的模型 LoRA 权重
    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    print(f"Training finished. Final LoRA model saved to {final_dir}")

    if tracker_initialized:
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.finish()
        elif args.use_swanlab and SWANLAB_AVAILABLE:
            swanlab.finish()


if __name__ == "__main__":
    main()
