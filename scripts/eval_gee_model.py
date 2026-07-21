import os
import re
import ast
import json
import time
import datetime
import argparse
import torch
import warnings
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

warnings.filterwarnings('ignore')

# 离线环境变量设置
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def extract_python_code(text):
    # Extract ```python ... ```
    pattern_py = re.compile(r'```python\s*(.*?)\s*```', re.DOTALL)
    matches_py = pattern_py.findall(text)
    if matches_py:
        return "\n".join(matches_py).strip()
        
    # Extract ``` ... ```
    pattern_generic = re.compile(r'```\s*(.*?)\s*```', re.DOTALL)
    matches_gen = pattern_generic.findall(text)
    if matches_gen:
        return "\n".join(matches_gen).strip()
        
    # Fallback to whole response
    return text.strip()

def check_syntax(code):
    if not code:
        return False
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False

def calc_api_accuracy(code, whitelist_set):
    if not code:
        return 0.0
        
    has_import = "import ee" in code or "from ee " in code
    has_init = "ee.Initialize" in code
    
    apis = re.findall(r'\bee\.([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)', code)
    if not apis:
        return 1.0 if (has_import and has_init) else 0.0
        
    valid_count = 0
    for api in apis:
        first_part = api.split('.')[0]
        if api in whitelist_set or f"ee.{api}" in whitelist_set or first_part in whitelist_set:
            valid_count += 1
            
    return valid_count / len(apis)

def tokenize_code(code):
    return re.findall(r'\w+|[^\w\s]', code)

def calc_rouge_l(reference, prediction):
    if not reference or not prediction:
        return 0.0
    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores['rougeL'].fmeasure
    except Exception:
        return 0.0

def calc_bleu_score(reference, prediction):
    if not reference or not prediction:
        return 0.0
    ref_tokens = tokenize_code(reference)
    pred_tokens = tokenize_code(prediction)
    if not ref_tokens or not pred_tokens:
        return 0.0
    try:
        chencherry = SmoothingFunction()
        return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=chencherry.method1)
    except Exception:
        return 0.0

def safe_calc_codebleu(reference, prediction):
    if not reference or not prediction:
        return 0.0
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu([[reference]], [prediction], lang="python")
        return result['codebleu']
    except Exception as e:
        # Fallback to simple BLEU score
        return calc_bleu_score(reference, prediction)

def load_whitelist(whitelist_path):
    whitelist = set()
    if os.path.exists(whitelist_path):
        with open(whitelist_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    whitelist.add(line)
        print(f"Loaded {len(whitelist)} APIs from whitelist.")
    else:
        print(f"Warning: Whitelist file {whitelist_path} not found.")
    return whitelist

def load_test_dataset(test_path):
    dataset = []
    if not os.path.exists(test_path):
        fallback_path = "data/gee_sft_dataset.jsonl"
        if os.path.exists(fallback_path):
            print(f"Warning: {test_path} not found. Fallback to {fallback_path} (using last 50 items for evaluation)")
            with open(fallback_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines[-50:]:
                    try:
                        data = json.loads(line)
                        dataset.append({
                            "prompt": data.get("instruction", ""),
                            "reference": data.get("output", "")
                        })
                    except Exception:
                        continue
            return dataset
        else:
            raise FileNotFoundError(f"Neither test file {test_path} nor fallback {fallback_path} exists.")
            
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                if "conversations" in data:
                    convs = data["conversations"]
                    prompt = ""
                    reference = ""
                    for c in convs:
                        if c["role"] == "user":
                            prompt = c["content"]
                        elif c["role"] == "assistant":
                            reference = c["content"]
                    dataset.append({"prompt": prompt, "reference": reference})
                elif "prompt" in data:
                    dataset.append({
                        "prompt": data["prompt"],
                        "reference": data.get("reference", data.get("output", ""))
                    })
                elif "instruction" in data:
                    dataset.append({
                        "prompt": data["instruction"],
                        "reference": data.get("output", "")
                    })
            except Exception:
                continue
    print(f"Loaded {len(dataset)} evaluation samples.")
    return dataset

def init_model(model_path, lora_path, device):
    # 优先检测本地 ModelScope 缓存或本地预下载路径
    modelscope_cache = os.path.expanduser("~/.cache/modelscope/hub/qwen/Qwen2.5-Coder-1.5B-Instruct")
    local_pretrained = "pretrained_models/Qwen/Qwen2.5-Coder-1.5B-Instruct"
    local_pretrained_alt = "pretrained_models/Qwen/Qwen2___5-Coder-1___5B-Instruct"
    
    if not os.path.exists(model_path) or model_path in ['./pretrained_models/Qwen2.5-Coder-1.5B-Instruct', 'Qwen/Qwen2.5-Coder-1.5B-Instruct', 'qwen/Qwen2.5-Coder-1.5B-Instruct']:
        if os.path.exists(modelscope_cache):
            print(f"Redirecting base model to local ModelScope cache: {modelscope_cache}")
            model_path = modelscope_cache
        elif os.path.exists(local_pretrained):
            print(f"Redirecting base model to local pre-downloaded path: {local_pretrained}")
            model_path = local_pretrained
        elif os.path.exists(local_pretrained_alt):
            print(f"Redirecting base model to local pre-downloaded path: {local_pretrained_alt}")
            model_path = local_pretrained_alt

    kwargs = {}
    if os.path.exists(model_path):
        kwargs["local_files_only"] = True
            
    print(f"Loading tokenizer from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, **kwargs)
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, **kwargs)
    
    if lora_path != 'None' and lora_path != '':
        print(f"Loading LoRA adapter from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)
        
    model = model.half().eval().to(device)
    return model, tokenizer

def update_report(report_path, model_name, metrics):
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    results = {}
    
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            match = re.search(r'<!-- EVAL_RESULTS_JSON\s*(.*?)\s*EVAL_RESULTS_JSON -->', content, re.DOTALL)
            if match:
                results = json.loads(match.group(1))
                print("Loaded existing results for comparison.")
        except Exception as e:
            print(f"Failed to parse existing report: {e}")
            
    results[model_name] = metrics
    
    for name in ['Base', 'SFT', 'GRPO']:
        if name not in results:
            results[name] = {
                "syntax_pass": 0.0,
                "api_acc": 0.0,
                "codebleu": 0.0,
                "rouge": 0.0,
                "samples": 0,
                "time": "-"
            }
            
    md_content = f"""# GEO-MiniMind 模型评估对比报表

本报表汇总了不同模型变体（Base, SFT, GRPO）在 Google Earth Engine Python API 代码生成测试集上的评测结果。

## 评估指标汇总

| 模型变体 | 语法通过率 (Syntax Pass Rate) | GEE API 准确率 (API Accuracy) | CodeBLEU / BLEU | ROUGE-L | 评估样本数 | 评估时间 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base** | {results['Base']['syntax_pass']:.2%} | {results['Base']['api_acc']:.2%} | {results['Base']['codebleu']:.2%} | {results['Base']['rouge']:.2%} | {results['Base']['samples']} | {results['Base']['time']} |
| **SFT** | {results['SFT']['syntax_pass']:.2%} | {results['SFT']['api_acc']:.2%} | {results['SFT']['codebleu']:.2%} | {results['SFT']['rouge']:.2%} | {results['SFT']['samples']} | {results['SFT']['time']} |
| **GRPO** | {results['GRPO']['syntax_pass']:.2%} | {results['GRPO']['api_acc']:.2%} | {results['GRPO']['codebleu']:.2%} | {results['GRPO']['rouge']:.2%} | {results['GRPO']['samples']} | {results['GRPO']['time']} |

*注：所有评估均在 GEE 测试集上运行批量推理，控制随机性参数为 temperature=0.2, top_p=0.9。*

## 指标解读与分析
- **语法通过率 (Syntax Pass Rate)**：衡量生成的 Python 代码是否符合基本语法规范，是否有明显的拼写或缩进错误。
- **GEE API 准确率 (API Accuracy)**：调用 API 是否在 GEE 官方及常用 API 白名单内。
- **CodeBLEU / BLEU**：与参考答案对比的代码级相似性得分，结合了语法树结构与关键字匹配。
- **ROUGE-L**：与参考答案的文本最长公共子序列匹配度。

<!-- EVAL_RESULTS_JSON
{json.dumps(results, indent=2)}
EVAL_RESULTS_JSON -->
"""
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Report successfully updated and saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate GEE models")
    parser.add_argument('--model_path', type=str, default='./pretrained_models/Qwen2.5-Coder-1.5B-Instruct', help="Base model path")
    parser.add_argument('--lora_path', type=str, default='None', help="LoRA adapter weight path")
    parser.add_argument('--model_name', type=str, default='Base', choices=['Base', 'SFT', 'GRPO'], help="Model variant name")
    parser.add_argument('--test_path', type=str, default='data/gee_sft_merged_test.jsonl', help="Path to test dataset jsonl")
    parser.add_argument('--whitelist_path', type=str, default='data/gee_api_whitelist.txt', help="Path to GEE API whitelist")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Running device")
    parser.add_argument('--num_samples', type=int, default=50, help="Number of samples to evaluate")
    parser.add_argument('--output_report', type=str, default='eval_results/eval_comparison.md', help="Output Markdown report path")
    args = parser.parse_args()
    
    whitelist = load_whitelist(args.whitelist_path)
    
    try:
        dataset = load_test_dataset(args.test_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
        
    if args.num_samples > 0:
        dataset = dataset[:args.num_samples]
    
    try:
        model, tokenizer = init_model(args.model_path, args.lora_path, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Note: If the model weights do not exist yet, you can still test the evaluation script infrastructure by updating dummy results.")
        raise e
        
    syntax_passes = 0
    api_accuracies = []
    codebleus = []
    rouges = []
    
    print(f"\nEvaluating {args.model_name} model on {len(dataset)} samples...")
    for item in tqdm(dataset):
        prompt = item["prompt"]
        ref_text = item["reference"]
        ref_code = extract_python_code(ref_text)
        
        messages = [
            {"role": "system", "content": "你是 GEE-Coder，一个专注于 Google Earth Engine Python API 的代码生成助手。"},
            {"role": "user", "content": prompt}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(args.device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
            
        prompt_len = inputs.input_ids.shape[1]
        response = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        pred_code = extract_python_code(response)
        
        syntax_ok = check_syntax(pred_code)
        if syntax_ok:
            syntax_passes += 1
            
        api_acc = calc_api_accuracy(pred_code, whitelist)
        api_accuracies.append(api_acc)
        
        cb_score = safe_calc_codebleu(ref_code, pred_code)
        codebleus.append(cb_score)
        
        r_score = calc_rouge_l(ref_code, pred_code)
        rouges.append(r_score)
        
    num_eval = len(dataset)
    avg_syntax = syntax_passes / num_eval if num_eval > 0 else 0.0
    avg_api = sum(api_accuracies) / num_eval if num_eval > 0 else 0.0
    avg_codebleu = sum(codebleus) / num_eval if num_eval > 0 else 0.0
    avg_rouge = sum(rouges) / num_eval if num_eval > 0 else 0.0
    
    print("\nEvaluation Results:")
    print(f"Syntax Pass Rate: {avg_syntax:.2%}")
    print(f"GEE API Accuracy: {avg_api:.2%}")
    print(f"CodeBLEU / BLEU: {avg_codebleu:.2%}")
    print(f"ROUGE-L: {avg_rouge:.2%}")
    
    metrics = {
        "syntax_pass": avg_syntax,
        "api_acc": avg_api,
        "codebleu": avg_codebleu,
        "rouge": avg_rouge,
        "samples": num_eval,
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    update_report(args.output_report, args.model_name, metrics)

if __name__ == '__main__':
    main()
