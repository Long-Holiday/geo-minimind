import re
import ast
import os

# 默认的 Mock 白名单，以防 data/gee_api_whitelist.txt 尚未就绪
MOCK_WHITELIST = {
    # 常用类与初始化
    "ee", "ee.Initialize", "ee.Image", "ee.ImageCollection", "ee.Feature", "ee.FeatureCollection",
    "ee.Geometry", "ee.Geometry.Point", "ee.Geometry.Polygon", "ee.Geometry.Rectangle", "ee.Geometry.BBox",
    "ee.Reducer", "ee.Reducer.mean", "ee.Reducer.sum", "ee.Reducer.min", "ee.Reducer.max", 
    "ee.Reducer.stdDev", "ee.Reducer.median", "ee.Reducer.count",
    "ee.Filter", "ee.Filter.date", "ee.Filter.bounds", "ee.Filter.eq", "ee.Filter.neq", 
    "ee.Filter.gt", "ee.Filter.gte", "ee.Filter.lt", "ee.Filter.lte", "ee.Filter.and", "ee.Filter.or",
    "ee.Date", "ee.List", "ee.Dictionary", "ee.Number", "ee.String", "ee.Algorithms", "ee.Join",
    # 常用实例方法（链式调用）
    "filterDate", "filterBounds", "filter", "select", "mean", "median", "sum", "min", "max", 
    "stdDev", "reduceRegion", "reduceRegions", "reduce", "clip", "subtract", "add", "multiply", 
    "divide", "normalizedDifference", "first", "toList", "getInfo", "map", "geometry", 
    "buffer", "slope", "aspect", "set", "get", "export", "visualize", "blend", "unmask", 
    "updateMask", "rename", "toBanded"
}

def load_whitelist():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    whitelist_path = os.path.join(current_dir, "..", "data", "gee_api_whitelist.txt")
    whitelist = set(MOCK_WHITELIST)
    if os.path.exists(whitelist_path):
        try:
            with open(whitelist_path, "r", encoding="utf-8") as f:
                for line in f:
                    symbol = line.strip()
                    if symbol:
                        whitelist.add(symbol)
        except Exception as e:
            print(f"Warning: Failed to load whitelist from {whitelist_path}: {e}")
    else:
        # 尝试使用家目录备用路径
        alt_path = os.path.expanduser("~/geo-minimind/data/gee_api_whitelist.txt")
        if os.path.exists(alt_path):
            try:
                with open(alt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        symbol = line.strip()
                        if symbol:
                            whitelist.add(symbol)
            except Exception:
                pass
    return whitelist

WHITELIST = load_whitelist()

# 预设的 GEE 关键词列表，用于 Completeness Score 提取
GEE_KEYWORDS_PRESET = {
    'ndvi', 'evi', 'reduceregion', 'beijing', 'shanghai', 'landsat', 'sentinel', 
    'mean', 'median', 'filterdate', 'filterbounds', 'clip', 'export', 'lst', 
    'dem', 'precipitation', 'temperature', 'water', 'forest', 'crop', 'cloud', 
    'mask', 'scale', 'crs', 'geometry', 'buffer', 'slope', 'aspect', 'interpolate', 
    'classify', 'randomforest', 'cart', 'svm', 'kmeans', 'spectral', 'unmix', 
    'chart', 'plot', 'print'
}

def extract_code(completion: str) -> str:
    """
    多级软降级提取 Markdown/Completion 中的 python 代码
    1. 标准 ```python ... ``` 块
    2. 通用 ``` ... ``` 块
    3. 未闭合的 ```python ... 或 ``` ...
    4. 软降级：完全没有 ``` 标记时，若文本包含 Python 代码特征（import, def, ee., =, print 等），提取代码行
    """
    if not completion or not completion.strip():
        return ""

    # 1. 尝试匹配完全闭合的 ```python ... ``` 块
    match = re.search(r"```python\s*(.*?)\s*```", completion, re.DOTALL)
    if match and match.group(1).strip():
        return match.group(1).strip()

    # 2. 尝试匹配完全闭合的 ``` ... ``` 块
    match = re.search(r"```\s*(.*?)\s*```", completion, re.DOTALL)
    if match and match.group(1).strip():
        return match.group(1).strip()

    # 3. 尝试匹配未闭合的 ```python ... 或 ``` ... (即模型生成到一半 truncated 或忘记写结尾 ```)
    match = re.search(r"```(?:python)?\s*(.*)", completion, re.DOTALL)
    if match and match.group(1).strip():
        return match.group(1).strip()

    # 4. 软降级：如果没有 ``` 标记，检查是否包含 Python/GEE 代码特征
    lines = completion.split("\n")
    code_lines = []
    code_keywords = ("import ", "from ", "ee.", "def ", "return ", "print(", "=", "dataset", "image", "collection", "filter")
    for line in lines:
        stripped = line.strip()
        # 如果整行包含典型代码特征，或者缩进大于等于4且非纯中文
        if any(kw in stripped for kw in code_keywords) or (line.startswith("    ") and not re.match(r"^[\u4e00-\u9fa5\s]+$", stripped)):
            code_lines.append(line)

    if code_lines:
        return "\n".join(code_lines).strip()

    return ""

def compute_syntax_score(code: str) -> float:
    """计算语法得分 (0.3)"""
    if not code:
        return 0.0
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError:
        # 即使 AST 语法报错，若代码中包含合法语句或结构，给予 0.2 的阶梯分，引导梯度
        has_basic_kw = bool(re.search(r"\b(import|from|def|ee\.[a-zA-Z0-9]+|=)\b", code))
        return 0.2 if has_basic_kw else 0.05
    except Exception:
        return 0.0

def compute_api_score(code: str) -> float:
    """计算 API 得分 (0.35)"""
    if not code:
        return 0.0
    
    # 检查是否包含 import ee 和 ee.Initialize()
    has_import = bool(re.search(r"\bimport\s+ee\b", code)) or ("ee" in code and "import" in code)
    has_init = "ee.Initialize" in code
    
    # 正则提取调用的 ee.xxx 符号及链式方法名
    ee_calls = re.findall(r"\b(ee\.[a-zA-Z0-9_\.]+)\b", code)
    method_calls = re.findall(r"\.([a-zA-Z0-9_]+)\s*\(", code)
    
    all_calls = []
    for call in ee_calls:
        all_calls.append(call)
        parts = call.split('.')
        if len(parts) > 2:
            all_calls.append(".".join(parts[:2]))
    
    all_calls.extend(method_calls)
    all_calls = [c for c in all_calls if c]
    
    if not all_calls:
        return 0.4 if (has_import and has_init) else (0.2 if has_import else 0.0)
    
    valid_count = 0
    for call in all_calls:
        if call in WHITELIST:
            valid_count += 1
        elif call.startswith("ee.") and call[3:] in WHITELIST:
            valid_count += 1
        elif f"ee.{call}" in WHITELIST:
            valid_count += 1
            
    compliance_rate = valid_count / len(all_calls)
    init_factor = 1.0 if (has_import and has_init) else (0.7 if has_import else 0.5)
    return compliance_rate * init_factor

def compute_format_score(completion: str) -> float:
    """计算格式得分 (0.15)，连续平滑判定"""
    if not completion:
        return 0.0

    has_std_py_block = bool(re.search(r"```python\s*(.*?)\s*```", completion, re.DOTALL))
    has_any_block = bool(re.search(r"```\s*(.*?)\s*```", completion, re.DOTALL))
    has_unclosed_block = bool(re.search(r"```(?:python)?\s*(.*)", completion, re.DOTALL))
    
    code = extract_code(completion)
    has_extracted_code = bool(code.strip())

    # 连续平滑长度因子 (在 100-800 字符范围保持高分，超出微幅平滑衰减，产生微小浮点分差)
    import math
    comp_len = len(completion)
    optimal_len = 350.0
    len_factor = math.exp(-((comp_len - optimal_len) / 600.0) ** 2) * 0.08

    if has_std_py_block:
        base = 0.92
    elif has_any_block:
        base = 0.75
    elif has_unclosed_block:
        base = 0.55
    elif has_extracted_code:
        base = 0.35
    else:
        base = 0.05

    return min(1.0, max(0.0, base + len_factor))

def extract_user_prompt(prompt: str) -> str:
    """从可能包含 chat template 的 prompt 字符串中提取出用户真实 prompt"""
    if "<|im_start|>user" in prompt:
        match = re.search(r"<\|im_start\|>user\n(.*?)\n<\|im_end\|>", prompt, re.DOTALL)
        if match and match.group(1) is not None:
            return match.group(1).strip()
        match = re.search(r"<\|im_start\|>user\s*(.*?)\s*<\|im_end\|>", prompt, re.DOTALL)
        if match and match.group(1) is not None:
            return match.group(1).strip()
    return prompt

def compute_completeness_score(prompt: str, code: str) -> float:
    """计算完成度得分 (0.2)"""
    if not code:
        return 0.0
        
    real_prompt = extract_user_prompt(prompt)
    keywords = set()
    
    backticks = re.findall(r"`([^`]+)`", real_prompt)
    for w in backticks:
        keywords.add(w.strip().lower())
        
    cap_words = re.findall(r"\b[A-Z][a-zA-Z0-9_]*\b", real_prompt)
    for w in cap_words:
        keywords.add(w.lower())
        
    camel_words = re.findall(r"\b[a-z]+[A-Z][a-zA-Z0-9_]*\b", real_prompt)
    for w in camel_words:
        keywords.add(w.lower())
        
    prompt_lower = real_prompt.lower()
    for w in GEE_KEYWORDS_PRESET:
        if w in prompt_lower:
            keywords.add(w)
            
    stop_words = {"the", "and", "for", "with", "from", "your", "code", "write", "model", "python", "earth", "engine"}
    keywords = {w for w in keywords if len(w) >= 3 and w not in stop_words}
    
    if not keywords:
        return 1.0
        
    code_lower = code.lower()
    matched_count = sum(1 for w in keywords if w in code_lower)
    return matched_count / len(keywords)

# TRL 兼容的 reward functions
def syntax_reward_func(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for prompt, completion in zip(prompts, completions):
        code = extract_code(completion)
        rewards.append(compute_syntax_score(code))
    return rewards

def api_reward_func(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for prompt, completion in zip(prompts, completions):
        code = extract_code(completion)
        rewards.append(compute_api_score(code))
    return rewards

def format_reward_func(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for prompt, completion in zip(prompts, completions):
        rewards.append(compute_format_score(completion))
    return rewards

def completeness_reward_func(prompts, completions, **kwargs) -> list[float]:
    rewards = []
    for prompt, completion in zip(prompts, completions):
        code = extract_code(completion)
        rewards.append(compute_completeness_score(prompt, code))
    return rewards

def gee_reward_func(prompts, completions, **kwargs) -> list[float]:
    """总分 = 0.3 * syntax_score + 0.35 * api_score + 0.15 * format_score + 0.2 * completeness_score + 微偏差项"""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        code = extract_code(completion)
        
        syntax = compute_syntax_score(code)
        api = compute_api_score(code)
        fmt = compute_format_score(completion)
        comp = compute_completeness_score(prompt, code)
        
        base_total = 0.3 * syntax + 0.35 * api + 0.15 * fmt + 0.2 * comp

        # 增加细腻的连续浮点偏置 (Fine-grained continuous bias)，保证不同生成的细节微调能体现区分度
        comp_len = len(completion)
        code_len = len(code)
        
        # 代码占总输出的比例（鼓励干货代码，减少无用废话）
        code_ratio = code_len / (comp_len + 1e-5)
        
        # 词汇丰富度与行数因子
        words = [w for w in re.findall(r"\w+", code.lower()) if w]
        unique_ratio = len(set(words)) / (len(words) + 1e-5) if words else 0.0
        
        non_empty_lines = [l for l in code.split("\n") if l.strip()]
        line_count_factor = min(len(non_empty_lines) / 25.0, 1.0)
        
        # 字符哈希微小扰动 (0 ~ 0.005)，即使代码结构相似，文本细节不同也会产生微小分差以激活 R_std 梯度
        char_nuance = 0.005 * (abs(hash(completion)) % 1000) / 1000.0 if comp_len > 0 else 0.0

        continuous_bias = 0.02 * code_ratio + 0.015 * unique_ratio + 0.01 * line_count_factor + char_nuance
        total = min(1.0, max(0.0, base_total + continuous_bias))
        rewards.append(total)
    return rewards

if __name__ == "__main__":
    print("=== Testing GEE Reward Function Optimizations ===")
    test_prompt = "Write a GEE script to compute NDVI for Beijing in 2023."
    
    test_cases = [
        ("Standard Markdown", "Here is the code:\n```python\nimport ee\nee.Initialize()\nimage = ee.Image('LANDSAT/LC08/C02/T1_TOA/LC08_044034_20140318')\nndvi = image.normalizedDifference(['B5', 'B4'])\nprint(ndvi.getInfo())\n```"),
        ("Unclosed Markdown", "Here is the code:\n```python\nimport ee\nee.Initialize()\nimage = ee.Image('LANDSAT/LC08/C02/T1_TOA/LC08_044034_20140318')\nndvi = image.normalizedDifference(['B5', 'B4'])"),
        ("No Markdown (Raw Code)", "import ee\nee.Initialize()\nimage = ee.Image('LANDSAT/LC08/C02/T1_TOA/LC08_044034_20140318')\nndvi = image.normalizedDifference(['B5', 'B4'])"),
        ("Broken Syntax Code", "```python\nimport ee\nimage = ee.Image(\n```"),
        ("Irrelevant Text", "I don't know how to write GEE Python script for NDVI.")
    ]

    for name, completion in test_cases:
        code = extract_code(completion)
        syntax = compute_syntax_score(code)
        api = compute_api_score(code)
        fmt = compute_format_score(completion)
        comp = compute_completeness_score(test_prompt, code)
        total = gee_reward_func([test_prompt], [completion])[0]
        
        print(f"\nCase: [{name}]")
        print(f"  Extracted Code snippet: {repr(code[:40])}...")
        print(f"  Syntax: {syntax:.2f} | API: {api:.2f} | Format: {fmt:.2f} | Completeness: {comp:.2f} => Total: {total:.4f}")
