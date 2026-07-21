import json
import os
import torch
from torch.utils.data import Dataset

class GEESFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))
                        
    def __len__(self):
        return len(self.samples)
        
    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        
        im_start_id = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        assistant_ids = self.tokenizer.encode("assistant", add_special_tokens=False)
        assistant_id = assistant_ids[0] if assistant_ids else 77091
        
        i = 0
        n = len(input_ids)
        while i < n - 2:
            if input_ids[i] == im_start_id and input_ids[i+1] == assistant_id:
                start = i + 3
                end = start
                while end < n:
                    if input_ids[end] == im_end_id:
                        break
                    end += 1
                for j in range(start, min(end + 1, n)):
                    labels[j] = input_ids[j]
                i = end + 1
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        conversations = sample['conversations']
        
        prompt = self.tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=False)
        
        encodings = self.tokenizer(
            prompt,
            max_length=self.max_length,
            truncation=True,
            padding="max_length"
        )
        
        input_ids = encodings.input_ids
        attention_mask = encodings.attention_mask
        labels = self.generate_labels(input_ids)
        
        # 确保被 pad 的部分 labels 也是 -100
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 151643
        for i in range(len(input_ids)):
            if input_ids[i] == pad_id:
                labels[i] = -100
                
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }

class GEERLDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        sample = self.samples[index]
        user_goal = sample.get('prompt', '')
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

if __name__ == '__main__':
    from transformers import AutoTokenizer
    print("Loading Qwen tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B-Instruct')
    # 确保 pad_token_id 存在
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print("Testing GEESFTDataset...")
    sft_dataset = GEESFTDataset('/home/default_user/geo-minimind/data/gee_sft_merged_val.jsonl', tokenizer, max_length=512)
    print(f"Loaded {len(sft_dataset)} validation samples.")
    if len(sft_dataset) > 0:
        item = sft_dataset[0]
        input_ids = item['input_ids'].tolist()
        labels = item['labels'].tolist()
        print("First sample token checking (first 80 tokens):")
        for i in range(min(80, len(input_ids))):
            token_str = tokenizer.decode([input_ids[i]])
            label_val = labels[i]
            print(f"{i:2d}: Token={repr(token_str):12s} ID={input_ids[i]:6d} Label={label_val}")
            
    print("\nTesting GEERLDataset...")
    rl_dataset = GEERLDataset('/home/default_user/geo-minimind/data/gee_rl_prompts_test.jsonl', tokenizer, max_length=512)
    print(f"Loaded {len(rl_dataset)} RL test prompts.")
    if len(rl_dataset) > 0:
        item = rl_dataset[0]
        print("First prompt sample:")
        print(repr(item['prompt']))

