import sys
import os

# 将根目录加入到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from trainer.gee_reward import (
    compute_syntax_score,
    compute_api_score,
    compute_format_score,
    compute_completeness_score,
    gee_reward_func,
    extract_user_prompt
)

def run_tests():
    print("=== Start GEE Reward Functions Unit Tests ===")
    
    # 1. 测试 Python 语法得分 (Syntax Score)
    print("\n[Test 1] Testing Syntax Score...")
    correct_py = "```python\nimport ee\nee.Initialize()\nimage = ee.Image('USGS/SRTMGL1_003')\nprint(image.getInfo())\n```"
    incorrect_py = "```python\nimport ee\nee.Initialize(\nimage = ee.Image('USGS/SRTMGL1_003'\n```"
    
    from trainer.gee_reward import extract_code
    code_correct = extract_code(correct_py)
    code_incorrect = extract_code(incorrect_py)
    
    score_correct = compute_syntax_score(code_correct)
    score_incorrect = compute_syntax_score(code_incorrect)
    
    print(f"Correct Python syntax score: {score_correct} (Expected: 1.0)")
    print(f"Incorrect Python syntax score: {score_incorrect} (Expected: 0.0)")
    assert score_correct == 1.0
    assert score_incorrect == 0.0
    
    # 2. 测试 API 得分 (API Score)
    print("\n[Test 2] Testing API Score...")
    # 有初始化，且均是白名单内的 API
    api_code_good = "import ee\nee.Initialize()\nimg = ee.Image('USGS/SRTMGL1_003')\nreduced = img.reduceRegion(reducer=ee.Reducer.mean())"
    # 有初始化，但调用了非法 API （假设 ee.NonExistentAPI 不在白名单里）
    api_code_bad = "import ee\nee.Initialize()\nimg = ee.Image('USGS/SRTMGL1_003')\nval = ee.NonExistentAPI('test')"
    # 缺少初始化
    api_code_no_init = "img = ee.Image('USGS/SRTMGL1_003')"
    
    score_api_good = compute_api_score(api_code_good)
    score_api_bad = compute_api_score(api_code_bad)
    score_api_no_init = compute_api_score(api_code_no_init)
    
    print(f"Good API score (all whitelisted & init): {score_api_good:.4f}")
    print(f"Bad API score (non-whitelisted used): {score_api_bad:.4f}")
    print(f"No init API score: {score_api_no_init:.4f}")
    
    assert score_api_good > score_api_bad
    assert score_api_bad > score_api_no_init
    
    # 3. 测试格式得分 (Format Score)
    print("\n[Test 3] Testing Format Score...")
    good_format = "Here is the code:\n```python\nimport ee\nee.Initialize()\n```\nHope this helps!" # 包含 ```python``` 且长度合适
    bad_format_too_short = "```python\nimport ee\n```" # 长度过短 (< 50 字符)
    bad_format_no_block = "import ee\nee.Initialize()\n" # 不包含 markdown 块
    
    score_fmt_good = compute_format_score(good_format)
    score_fmt_short = compute_format_score(bad_format_too_short)
    score_fmt_no_block = compute_format_score(bad_format_no_block)
    
    print(f"Good format score (block & length): {score_fmt_good} (Expected: 1.0)")
    print(f"Too short format score: {score_fmt_short} (Expected: 0.5 or less)")
    print(f"No block format score: {score_fmt_no_block} (Expected: 0.0)")
    
    assert score_fmt_good == 1.0
    assert score_fmt_short < 1.0
    assert score_fmt_no_block == 0.0
    
    # 4. 测试完成度得分 (Completeness Score)
    print("\n[Test 4] Testing Completeness Score...")
    prompt = "Write a GEE script to calculate `NDVI` for Beijing region and compute the mean."
    matching_code = "import ee\nee.Initialize()\nndvi = image.normalizedDifference(['B5', 'B4'])\nmean_val = ndvi.reduceRegion(reducer=ee.Reducer.mean(), geometry=Beijing)"
    non_matching_code = "import ee\nee.Initialize()\nx = 1 + 2\nprint(x)"
    
    score_comp_good = compute_completeness_score(prompt, matching_code)
    score_comp_bad = compute_completeness_score(prompt, non_matching_code)
    
    print(f"Matching prompt completeness score: {score_comp_good:.4f}")
    print(f"Non-matching prompt completeness score: {score_comp_bad:.4f}")
    
    assert score_comp_good > score_comp_bad
    
    # 5. 测试用户 Prompt 提取
    print("\n[Test 5] Testing User Prompt Extractor...")
    chat_prompt = "<|im_start|>system\nYou are GEE-Coder<|im_end|>\n<|im_start|>user\nWrite a GEE script for NDVI.<|im_end|>\n<|im_start|>assistant\n"
    extracted = extract_user_prompt(chat_prompt)
    print(f"Original Chat Prompt: {repr(chat_prompt)}")
    print(f"Extracted User Prompt: {repr(extracted)}")
    assert extracted == "Write a GEE script for NDVI."
    
    # 6. 测试总得分加权计算
    print("\n[Test 6] Testing Combined Reward Function...")
    prompts = [prompt]
    completions = [good_format]
    total_rewards = gee_reward_func(prompts, completions)
    print(f"Combined total reward for sample: {total_rewards[0]:.4f}")
    assert 0.0 <= total_rewards[0] <= 1.0
    
    print("\n=== All Unit Tests Passed Successfully! ===")

if __name__ == "__main__":
    run_tests()
