import json
import os
import re
import random

random.seed(42)

def augment_prompt(user_content):
    # 如果 prompt 已经很长或者开头已经有某种前缀，可以随机改变它
    prefixes = [
        "请使用 GEE Python API 实现：",
        "在 Google Earth Engine 中，如何用 Python API 完成：",
        "使用 Earth Engine Python 接口，请帮我编写代码以：",
        "请帮我用 GEE Python 解决以下问题：",
        "基于 Google Earth Engine Python API："
    ]
    # 清理掉可能存在的开头常用引导语，避免引导语重复
    cleaned = user_content
    strip_patterns = [
        r"^在\s*Google Earth Engine\s*\(GEE\)\s*Python API\s*中，请解释",
        r"^在\s*Google Earth Engine\s*Python API\s*中，如何使用",
        r"^在\s*Google Earth Engine\s*Python API\s*中，",
        r"^请使用\s*GEE\s*Python\s*API，",
        r"^使用\s*GEE\s*Python\s*API，"
    ]
    for pattern in strip_patterns:
        cleaned = re.sub(pattern, "", cleaned)
    cleaned = cleaned.lstrip(" ，,。.")
    
    return random.choice(prefixes) + cleaned

def augment_code_comments(assistant_content):
    # 查找代码块
    code_blocks = re.findall(r"```python\n(.*?)```", assistant_content, re.DOTALL)
    if not code_blocks:
        return assistant_content
        
    augmented_content = assistant_content
    for code in code_blocks:
        new_code = code
        # 简单替换一些常见的注释
        replacements = {
            "# 调用核心方法": "# 执行 GEE 核心 API 调用",
            "# 运行核心方法": "# 触发 Earth Engine 核心计算",
            "import ee\nee.Initialize()": "# 导入并初始化 Earth Engine\nimport ee\nee.Initialize()",
            "import ee\nimport ee.mapclient\n\nee.Initialize()": "import ee\nimport ee.mapclient\n# 初始化 Earth Engine 服务\nee.Initialize()",
            "# 核心逻辑": "# GEE 核心处理逻辑",
            "# 打印结果": "# 输出计算结果"
        }
        for old_comment, new_comment in replacements.items():
            if old_comment in new_code:
                new_code = new_code.replace(old_comment, new_comment)
        
        # 替换回原内容
        augmented_content = augmented_content.replace(code, new_code)
        
    return augmented_content

def process_sft_data():
    sft_dataset_path = '/home/default_user/geo-minimind/data/gee_sft_dataset.jsonl'
    arl_files = [
        '/home/default_user/geo-minimind/data/arl_sft_converted.jsonl',
        '/home/default_user/geo-minimind/data/arl_api_complex.jsonl',
        '/home/default_user/geo-minimind/data/arl_examples_debug.jsonl'
    ]
    output_sft_path = '/home/default_user/geo-minimind/data/gee_sft_merged.jsonl'
    
    merged_data = []
    
    # 1. 读 gee_sft_dataset.jsonl
    if os.path.exists(sft_dataset_path):
        with open(sft_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                user = item.get('instruction', '').strip()
                assistant = item.get('output', '').strip()
                if user and assistant:
                    # 原始样本
                    conv = {
                        "conversations": [
                            {"role": "system", "content": "你是 GEE-Coder，一个专注于 Google Earth Engine Python API 的代码生成助手。"},
                            {"role": "user", "content": user},
                            {"role": "assistant", "content": assistant}
                        ]
                    }
                    merged_data.append(conv)
                    
                    # 数据增强样本 1: 改变 user 的前缀
                    aug_user = augment_prompt(user)
                    conv_aug1 = {
                        "conversations": [
                            {"role": "system", "content": "你是 GEE-Coder，一个专注于 Google Earth Engine Python API 的代码生成助手。"},
                            {"role": "user", "content": aug_user},
                            {"role": "assistant", "content": assistant}
                        ]
                    }
                    merged_data.append(conv_aug1)
                    
                    # 数据增强样本 2: 改变代码注释（如果包含代码）
                    aug_assistant = augment_code_comments(assistant)
                    if aug_assistant != assistant:
                        conv_aug2 = {
                            "conversations": [
                                {"role": "system", "content": "你是 GEE-Coder，一个专注于 Google Earth Engine Python API 的代码生成助手。"},
                                {"role": "user", "content": user},
                                {"role": "assistant", "content": aug_assistant}
                            ]
                        }
                        merged_data.append(conv_aug2)
                        
    # 2. 读 arl_*.jsonl
    for file_path in arl_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    user = item.get('user_goal', '').strip()
                    assistant = item.get('final_answer', '').strip()
                    if user and assistant:
                        # 原始样本
                        conv = {
                            "conversations": [
                                {"role": "system", "content": "你是 GEE-Coder，一个专注于 Google Earth Engine Python API 的代码生成助手。"},
                                {"role": "user", "content": user},
                                {"role": "assistant", "content": assistant}
                            ]
                        }
                        merged_data.append(conv)
                        
                        # 增强样本 1: 改变 user 的前缀
                        aug_user = augment_prompt(user)
                        conv_aug1 = {
                            "conversations": [
                                {"role": "system", "content": "你是 GEE-Coder，一个专注于 Google Earth Engine Python API 的代码生成助手。"},
                                {"role": "user", "content": aug_user},
                                {"role": "assistant", "content": assistant}
                            ]
                        }
                        merged_data.append(conv_aug1)
                        
                        # 增强样本 2: 改变代码注释（如果包含代码）
                        aug_assistant = augment_code_comments(assistant)
                        if aug_assistant != assistant:
                            conv_aug2 = {
                                "conversations": [
                                    {"role": "system", "content": "你是 GEE-Coder，一个专注于 Google Earth Engine Python API 的代码生成助手。"},
                                    {"role": "user", "content": user},
                                    {"role": "assistant", "content": aug_assistant}
                                ]
                            }
                            merged_data.append(conv_aug2)
                            
    # 写入 gee_sft_merged.jsonl
    os.makedirs(os.path.dirname(output_sft_path), exist_ok=True)
    with open(output_sft_path, 'w', encoding='utf-8') as f:
        for item in merged_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Generated {len(merged_data)} samples in {output_sft_path}")

def process_rl_data():
    rl_dataset_path = '/home/default_user/geo-minimind/data/gee_agentic_rl_dataset.jsonl'
    output_rl_path = '/home/default_user/geo-minimind/data/gee_rl_prompts.jsonl'
    
    rl_prompts = []
    if os.path.exists(rl_dataset_path):
        with open(rl_dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                user_goal = item.get('user_goal', '').strip()
                if user_goal:
                    rl_prompts.append({"prompt": user_goal})
                    
    # 写入 gee_rl_prompts.jsonl
    os.makedirs(os.path.dirname(output_rl_path), exist_ok=True)
    with open(output_rl_path, 'w', encoding='utf-8') as f:
        for item in rl_prompts:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Generated {len(rl_prompts)} prompts in {output_rl_path}")

if __name__ == '__main__':
    process_sft_data()
    process_rl_data()
