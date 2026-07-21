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
        # 尝试使用绝对路径 /home/default_user/geo-minimind/data/gee_api_whitelist.txt
        alt_path = "/home/default_user/geo-minimind/data/gee_api_whitelist.txt"
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
    """提取 Markdown 中的 python 代码块"""
    match = re.search(r"```python\s*(.*?)\s*```", completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"```\s*(.*?)\s*```", completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def compute_syntax_score(code: str) -> float:
    """计算语法得分 (0.3)"""
    if not code:
        return 0.0
    try:
        ast.parse(code)
        return 1.0
    except SyntaxError:
        return 0.0
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
        return 0.5 if (has_import and has_init) else 0.0
    
    valid_count = 0
    for call in all_calls:
        if call in WHITELIST:
            valid_count += 1
        elif call.startswith("ee.") and call[3:] in WHITELIST:
            valid_count += 1
        elif f"ee.{call}" in WHITELIST:
            valid_count += 1
            
    compliance_rate = valid_count / len(all_calls)
    init_factor = 1.0 if (has_import and has_init) else 0.5
    return compliance_rate * init_factor

def compute_format_score(completion: str) -> float:
    """计算格式得分 (0.15)"""
    has_py_block = bool(re.search(r"```python\s*(.*?)\s*```", completion, re.DOTALL))
    correct_length = 50 <= len(completion) <= 1000
    
    if has_py_block and correct_length:
        return 1.0
    elif has_py_block or correct_length:
        return 0.5
    return 0.0

def extract_user_prompt(prompt: str) -> str:
    """从可能包含 chat template 的 prompt 字符串中提取出用户真实 prompt"""
    if "<|im_start|>user" in prompt:
        # 对 | 进行转义，防范逻辑或 OR 元字符混淆
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
    """总分 = 0.3 * syntax_score + 0.35 * api_score + 0.15 * format_score + 0.2 * completeness_score"""
    rewards = []
    for prompt, completion in zip(prompts, completions):
        code = extract_code(completion)
        
        syntax = compute_syntax_score(code)
        api = compute_api_score(code)
        fmt = compute_format_score(completion)
        comp = compute_completeness_score(prompt, code)
        
        total = 0.3 * syntax + 0.35 * api + 0.15 * fmt + 0.2 * comp
        rewards.append(total)
    return rewards
