import json
import os
import random

def split_sft_dataset():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_path = os.path.join(project_root, 'data', 'gee_sft_merged.jsonl')
    train_path = os.path.join(project_root, 'data', 'gee_sft_merged_train.jsonl')
    val_path = os.path.join(project_root, 'data', 'gee_sft_merged_val.jsonl')
    test_path = os.path.join(project_root, 'data', 'gee_sft_merged_test.jsonl')
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return
        
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
        
    random.seed(42)
    random.shuffle(lines)
    
    total = len(lines)
    train_end = int(total * 0.8)
    val_end = train_end + int(total * 0.1)
    
    train_lines = lines[:train_end]
    val_lines = lines[train_end:val_end]
    test_lines = lines[val_end:]
    
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(val_path, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    with open(test_path, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
        
    print(f"SFT Dataset Split: Total={total}, Train={len(train_lines)}, Val={len(val_lines)}, Test={len(test_lines)}")

def split_rl_dataset():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    input_path = os.path.join(project_root, 'data', 'gee_rl_prompts.jsonl')
    train_path = os.path.join(project_root, 'data', 'gee_rl_prompts_train.jsonl')
    test_path = os.path.join(project_root, 'data', 'gee_rl_prompts_test.jsonl')
    
    if not os.path.exists(input_path):
        print(f"File not found: {input_path}")
        return
        
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line for line in f if line.strip()]
        
    random.seed(42)
    random.shuffle(lines)
    
    total = len(lines)
    train_end = int(total * 0.8)
    
    train_lines = lines[:train_end]
    test_lines = lines[train_end:]
    
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    with open(test_path, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
        
    print(f"RL Dataset Split: Total={total}, Train={len(train_lines)}, Test={len(test_lines)}")

if __name__ == '__main__':
    split_sft_dataset()
    split_rl_dataset()
