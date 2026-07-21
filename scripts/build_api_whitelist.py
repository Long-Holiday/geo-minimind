import json
import os

def build_whitelist():
    dataset_path = '/home/default_user/geo-minimind/data/gee_sft_dataset.jsonl'
    output_path = '/home/default_user/geo-minimind/data/gee_api_whitelist.txt'
    
    symbols = set()
    
    # 基础的 GEE 常规 API
    common_gee_apis = [
        "ee.Image", "ee.ImageCollection", "ee.Feature", "ee.FeatureCollection",
        "ee.Geometry", "ee.Date", "ee.List", "ee.Dictionary", "ee.Number",
        "ee.String", "ee.Reducer", "ee.Projection", "ee.Filter", "ee.Join",
        "ee.Kernel", "ee.Map", "ee.Algorithms", "ee.Clusterer", "ee.Classifier",
        "ee.Model", "ee.Export"
    ]
    for api in common_gee_apis:
        symbols.add(api)
        # 加上不带 ee. 的版本
        if api.startswith("ee."):
            symbols.add(api[3:])
            
    if os.path.exists(dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    metadata = data.get('metadata', {})
                    api_symbol = metadata.get('api_symbol')
                    if api_symbol:
                        # 原样添加
                        symbols.add(api_symbol)
                        # 添加带有 ee. 前缀的
                        if not api_symbol.startswith("ee."):
                            symbols.add(f"ee.{api_symbol}")
                            
                        # 处理 __init__ 的情况
                        if api_symbol.endswith(".__init__"):
                            base = api_symbol[:-9]
                            symbols.add(base)
                            symbols.add(f"ee.{base}")
                        # 处理可能包含 of ee.
                        if api_symbol.startswith("ee."):
                            symbols.add(api_symbol[3:])
                except Exception as e:
                    print(f"Error parsing line: {e}")
                    
    sorted_symbols = sorted(list(symbols))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for s in sorted_symbols:
            f.write(f"{s}\n")
            
    print(f"Generated {len(sorted_symbols)} symbols in {output_path}")

if __name__ == '__main__':
    build_whitelist()
