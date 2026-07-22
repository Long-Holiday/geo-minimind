#!/usr/bin/env bash

# ==============================================================================
# GEO-MiniMind 一键启动管理脚本
# ==============================================================================

# ANSI Color 终端色彩定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# 自动定位项目根目录
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_ROOT" || exit 1

# 显示 ASCII Banner 艺术字
show_banner() {
    clear
    echo -e "${CYAN}${BOLD}"
    echo "   ______ ______ ____        __  ___ _         _ __  ___ _             __ "
    echo "  / ____// ____// __ \      /  |/  /(_)____   (_)  |/  /(_)____   ____/ / "
    echo " / / __ / __/  / / / /____ / /|_/ // // __ \ / // /|_/ // // __ \ / __  /  "
    echo "/ /_/ // /___ / /_/ //___// /  / // // / / // // /  / // // / / // /_/ /   "
    echo "\____//_____/ \____/     /_/  /_//_//_/ /_//_//_/  /_//_//_/ /_/ \__,_/    "
    echo -e "${NC}"
    echo -e "${BLUE}======================================================================${NC}"
    echo -e "       ${BOLD}GEO-MiniMind 领域大模型管理工具 (Qwen2.5-Coder 微调后训练RL)${NC}"
    echo -e "       项目根目录: ${YELLOW}$PROJECT_ROOT${NC}"
    echo -e "${BLUE}======================================================================${NC}"
}

# 检测 Python 环境和显卡信息
check_environment() {
    echo -e "${CYAN}[Env Check] 正在检测运行环境...${NC}"
    
    # 智能优先选择 minimind 专用的 conda 虚拟环境
    if [ -x "$HOME/miniconda3/envs/minimind/bin/python" ]; then
        PYTHON_EXE="$HOME/miniconda3/envs/minimind/bin/python"
    elif [ -n "$CONDA_PREFIX" ] && [ -x "$CONDA_PREFIX/envs/minimind/bin/python" ]; then
        PYTHON_EXE="$CONDA_PREFIX/envs/minimind/bin/python"
    elif command -v python &> /dev/null; then
        PYTHON_EXE="python"
    elif command -v python3 &> /dev/null; then
        PYTHON_EXE="python3"
    else
        echo -e "${RED}[Error] 未找到 python 解释器，请检查 Python 是否已安装且在环境变量中。${NC}"
        return 1
    fi
    
    # 验证 python 是否可用
    if ! type "$PYTHON_EXE" &> /dev/null; then
        echo -e "${RED}[Error] 未找到 python 解释器，请检查 Python 是否已安装且在环境变量中。${NC}"
        return 1
    fi
    
    PYTHON_VER=$($PYTHON_EXE -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo -e "  - Python 解释器: ${GREEN}$($PYTHON_EXE -c 'import sys; print(sys.executable)')${NC} (v$PYTHON_VER)"
    
    # 验证 PyTorch 和 CUDA 显卡
    $PYTHON_EXE -c 'import torch' &> /dev/null
    if [ $? -eq 0 ]; then
        CUDA_AVAILABLE=$($PYTHON_EXE -c 'import torch; print(torch.cuda.is_available())')
        if [ "$CUDA_AVAILABLE" = "True" ]; then
            GPU_NAME=$($PYTHON_EXE -c 'import torch; print(torch.cuda.get_device_name(0))')
            echo -e "  - GPU 显卡: ${GREEN}$GPU_NAME${NC} (CUDA 可用)"
        else
            echo -e "  - GPU 显卡: ${YELLOW}未检测到 CUDA 显卡，将回退到 CPU 运行。${NC}"
        fi
    else
        echo -e "  - PyTorch 库: ${RED}未检测到 PyTorch，请确保您的 Python 环境已正确配置依赖包！${NC}"
    fi

    # 兼容处理 config.yaml / config.ymal
    if [ ! -f "config.yaml" ] && [ -f "config.ymal" ]; then
        echo -e "  - 配置文件: 找到 ${YELLOW}config.ymal${NC}，已软链接为标准 config.yaml"
        ln -sf config.ymal config.yaml
    elif [ -f "config.yaml" ]; then
        echo -e "  - 配置文件: 找到 ${GREEN}config.yaml${NC}"
    else
        echo -e "  - 配置文件: ${RED}未找到 config.yaml，将使用默认配置。${NC}"
    fi
    echo -e "${BLUE}----------------------------------------------------------------------${NC}"
}

# 0. 一键全流程自动化运行 (数据处理 -> SFT微调 -> 权重合并 -> GRPO对齐 -> 权重合并 -> 评估)
run_all_pipeline() {
    echo -e "${PURPLE}${BOLD}[Run All] 开始准备一键全流程自动化训练与评估...${NC}"
    echo -e "${YELLOW}全流程包含步骤：${NC}"
    echo -e "  1. [Data Prep] 数据格式转换与划分"
    echo -e "  2. [SFT Train] SFT LoRA 微调训练"
    echo -e "  3. [Merge SFT] 合并 SFT LoRA 权重到 out/qwen_sft_merged"
    echo -e "  4. [GRPO Train] GRPO 强化学习对齐训练"
    echo -e "  5. [Merge GRPO] 合并 GRPO LoRA 权重到 out/qwen_grpo_merged"
    echo -e "  6. [Evaluate] 运行三阶段模型对比评估"
    echo -e "${BLUE}----------------------------------------------------------------------${NC}"
    read -p "是否立即开始执行全流程？(y/N): " confirm_all
    if [[ ! "$confirm_all" =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}[Cancelled] 已取消全流程运行。${NC}"
        return 0
    fi

    # 步骤 1: 数据准备
    echo -e "\n${CYAN}=================== [Stage 1/6] 数据格式转换与划分 ===================${NC}"
    run_data_prep
    if [ $? -ne 0 ]; then
        echo -e "${RED}[Error] 步骤 1 数据准备失败，中止全流程。${NC}"
        return 1
    fi

    # 步骤 2: SFT 微调
    echo -e "\n${CYAN}=================== [Stage 2/6] SFT LoRA 微调训练 ===================${NC}"
    if [ -d "pretrained_models/Qwen/Qwen2.5-Coder-1.5B-Instruct" ]; then
        echo -e "${GREEN}[Model Check] 找到本地基座模型: pretrained_models/Qwen/Qwen2.5-Coder-1.5B-Instruct${NC}"
    elif [ -d "pretrained_models/Qwen/Qwen2___5-Coder-1___5B-Instruct" ]; then
        echo -e "${GREEN}[Model Check] 找到本地基座模型: pretrained_models/Qwen/Qwen2___5-Coder-1___5B-Instruct${NC}"
    elif [ -d "$HOME/.cache/modelscope/hub/qwen/Qwen2.5-Coder-1.5B-Instruct" ]; then
        echo -e "${GREEN}[Model Check] 找到 ModelScope 本地缓存基座模型${NC}"
    else
        echo -e "${YELLOW}[Model Check] 未找到本地基座模型目录，训练启动时将自动联网下载 (ModelScope / HuggingFace)...${NC}"
    fi
    echo -e "${CYAN}正在启动 train_qwen_lora.py...${NC}"
    $PYTHON_EXE trainer/train_qwen_lora.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}[Error] 步骤 2 SFT 训练失败，中止全流程。${NC}"
        return 1
    fi

    # 步骤 3: 合并 SFT 权重
    echo -e "\n${CYAN}=================== [Stage 3/6] 合并 SFT LoRA 权重 ===================${NC}"
    if [ ! -d "out/qwen_lora_sft" ]; then
        echo -e "${RED}[Error] 未找到 SFT LoRA 权重目录 (out/qwen_lora_sft)，中止全流程。${NC}"
        return 1
    fi
    echo -e "${CYAN}正在合并 SFT 权重到 out/qwen_sft_merged...${NC}"
    $PYTHON_EXE scripts/merge_lora_weights.py \
        --lora_model out/qwen_lora_sft \
        --output_dir out/qwen_sft_merged
    if [ $? -ne 0 ]; then
        echo -e "${RED}[Error] 步骤 3 SFT 权重合并失败，中止全流程。${NC}"
        return 1
    fi

    # 步骤 4: GRPO 强化学习对齐
    echo -e "\n${CYAN}=================== [Stage 4/6] GRPO 强化对齐训练 ===================${NC}"
    echo -e "${CYAN}正在启动 train_qwen_grpo.py...${NC}"
    $PYTHON_EXE trainer/train_qwen_grpo.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}[Error] 步骤 4 GRPO 训练失败，中止全流程。${NC}"
        return 1
    fi

    # 步骤 5: 合并 GRPO 权重
    echo -e "\n${CYAN}=================== [Stage 5/6] 合并 GRPO LoRA 权重 ===================${NC}"
    if [ ! -d "out/qwen_grpo/final" ]; then
        echo -e "${RED}[Error] 未找到 GRPO 训练最终权重 (out/qwen_grpo/final)，中止全流程。${NC}"
        return 1
    fi
    echo -e "${CYAN}正在合并 GRPO 权重到 out/qwen_grpo_merged...${NC}"
    $PYTHON_EXE scripts/merge_lora_weights.py \
        --lora_model out/qwen_grpo/final \
        --output_dir out/qwen_grpo_merged
    if [ $? -ne 0 ]; then
        echo -e "${RED}[Error] 步骤 5 GRPO 权重合并失败，中止全流程。${NC}"
        return 1
    fi

    # 步骤 6: 模型评估
    echo -e "\n${CYAN}=================== [Stage 6/6] 运行模型评估 ===================${NC}"
    run_evaluate

    echo -e "\n${GREEN}${BOLD}[Success] 🎉 恭喜！GEO-MiniMind 一键全流程训练与评估已全部完成！${NC}"
}

# 1. 一键完成数据转换与划分
run_data_prep() {
    echo -e "${CYAN}[Step 1] 开始执行数据格式转换...${NC}"
    if [ -f "scripts/convert_data_for_qwen.py" ]; then
        $PYTHON_EXE scripts/convert_data_for_qwen.py
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}[Success] 数据转换完成。${NC}"
        else
            echo -e "${RED}[Failed] 数据转换失败，请检查报错日志！${NC}"
            return 1
        fi
    else
        echo -e "${RED}[Error] 未找到 scripts/convert_data_for_qwen.py 文件！${NC}"
        return 1
    fi

    echo -e "${CYAN}[Step 2] 开始对转换后的数据集进行划分...${NC}"
    if [ -f "scripts/split_dataset.py" ]; then
        $PYTHON_EXE scripts/split_dataset.py
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}[Success] 数据集划分完成！${NC}"
        else
            echo -e "${RED}[Failed] 数据集划分失败，请检查报错日志！${NC}"
            return 1
        fi
    else
        echo -e "${RED}[Error] 未找到 scripts/split_dataset.py 文件！${NC}"
        return 1
    fi
    
    echo -e "${GREEN}[Done] 恭喜！数据转换与划分全流程执行成功。${NC}"
}

# 2. 启动 SFT LoRA 微调
run_sft_train() {
    echo -e "${CYAN}[SFT Train] 正在检查 SFT 训练数据和基座模型...${NC}"
    
    if [ ! -f "data/gee_sft_merged_train.jsonl" ]; then
        echo -e "${YELLOW}[Warning] 未找到 SFT 训练集文件 (data/gee_sft_merged_train.jsonl)。${NC}"
        read -p "是否需要立即运行 [Data Prep] 进行数据预处理？(y/N): " prep_choice
        if [[ "$prep_choice" =~ ^[Yy]$ ]]; then
            run_data_prep
        else
            echo -e "${RED}[Error] 训练被用户中止。${NC}"
            return 1
        fi
    fi

    # 检查本地 Qwen 目录或自动联网下载
    if [ -d "pretrained_models/Qwen/Qwen2.5-Coder-1.5B-Instruct" ]; then
        echo -e "${GREEN}[Model Check] 找到本地基座模型: pretrained_models/Qwen/Qwen2.5-Coder-1.5B-Instruct${NC}"
    elif [ -d "pretrained_models/Qwen/Qwen2___5-Coder-1___5B-Instruct" ]; then
        echo -e "${GREEN}[Model Check] 找到本地基座模型: pretrained_models/Qwen/Qwen2___5-Coder-1___5B-Instruct${NC}"
    elif [ -d "$HOME/.cache/modelscope/hub/qwen/Qwen2.5-Coder-1.5B-Instruct" ]; then
        echo -e "${GREEN}[Model Check] 找到 ModelScope 本地缓存基座模型${NC}"
    else
        echo -e "${YELLOW}[Model Check] 未检测到本地基座模型，训练启动时将自动联网下载 (ModelScope / HuggingFace)...${NC}"
    fi

    # 提供断点续训选项
    read -p "是否从上一个最近的 Checkpoint 恢复训练？(y/N, 默认 N): " resume_choice
    EXTRA_ARGS=""
    if [[ "$resume_choice" =~ ^[Yy]$ ]]; then
        EXTRA_ARGS="--from_resume"
        echo -e "${YELLOW}已启用断点续训模式。${NC}"
    fi

    echo -e "${CYAN}[SFT Train] 正在启动 train_qwen_lora.py... (配置读取自 config.yaml)${NC}"
    $PYTHON_EXE trainer/train_qwen_lora.py $EXTRA_ARGS
}

# 3. 合并 LoRA 权重
run_merge_weights() {
    echo -e "${CYAN}[Merge Weights] 权重合并管理子菜单：${NC}"
    echo "  1) 合并 SFT LoRA 权重 (out/qwen_lora_sft -> out/qwen_sft_merged)"
    echo "  2) 合并 GRPO LoRA 权重 (out/qwen_grpo/final -> out/qwen_grpo_merged)"
    echo "  3) 返回主菜单"
    read -p "请选择需要合并的对象 [1-3]: " merge_choice
    
    case $merge_choice in
        1)
            if [ ! -d "out/qwen_lora_sft" ]; then
                echo -e "${RED}[Error] 未找到 SFT 微调 LoRA 目录: out/qwen_lora_sft。请先进行 SFT 微调训练。${NC}"
                return 1
            fi
            echo -e "${CYAN}正在合并 SFT 权重到 out/qwen_sft_merged...${NC}"
            $PYTHON_EXE scripts/merge_lora_weights.py \
                --lora_model out/qwen_lora_sft \
                --output_dir out/qwen_sft_merged
            ;;
        2)
            if [ ! -d "out/qwen_grpo/final" ]; then
                echo -e "${RED}[Error] 未找到 GRPO 训练最终权重: out/qwen_grpo/final。请先完成 GRPO 训练。${NC}"
                return 1
            fi
            echo -e "${CYAN}正在合并 GRPO 权重到 out/qwen_grpo_merged...${NC}"
            $PYTHON_EXE scripts/merge_lora_weights.py \
                --lora_model out/qwen_grpo/final \
                --output_dir out/qwen_grpo_merged
            ;;
        3)
            return 0
            ;;
        *)
            echo -e "${RED}无效选择，返回主菜单。${NC}"
            ;;
    esac
}

# 4. 启动 GRPO 强化学习对齐
run_grpo_train() {
    echo -e "${CYAN}[GRPO Train] 正在检查对齐训练前置条件...${NC}"
    if [ ! -f "data/gee_rl_prompts_train.jsonl" ]; then
        echo -e "${YELLOW}[Warning] 未找到 RL 提示词训练集。${NC}"
        read -p "是否需要立即运行 [Data Prep] 进行数据预处理？(y/N): " prep_choice
        if [[ "$prep_choice" =~ ^[Yy]$ ]]; then
            run_data_prep
        else
            echo -e "${RED}[Error] 训练被用户中止。${NC}"
            return 1
        fi
    fi

    # 检测并提示使用合并后的 SFT 模型，如果不存在则警告
    if [ ! -d "out/qwen_sft_merged" ]; then
        echo -e "${YELLOW}[Warning] 未检测到已合并权重的 SFT 模型 (out/qwen_sft_merged)。${NC}"
        echo -e "按照 RL 后训练流程，GRPO 通常需要从 SFT 微调后的模型上进行对齐。"
        read -p "是否强制回退使用 Base 模型开始 GRPO 对齐？(y/N): " force_choice
        if [[ ! "$force_choice" =~ ^[Yy]$ ]]; then
            echo -e "${RED}[Cancelled] 请先选择 [Option 3] 将 SFT 权重与 Base 模型进行合并！${NC}"
            return 1
        fi
    fi

    read -p "是否从上一个最近的 GRPO Checkpoint 恢复训练？(y/N, 默认 N): " resume_choice
    EXTRA_ARGS=""
    if [[ "$resume_choice" =~ ^[Yy]$ ]]; then
        EXTRA_ARGS="--from_resume"
        echo -e "${YELLOW}已启用 GRPO 断点续训模式。${NC}"
    fi

    echo -e "${CYAN}[GRPO Train] 正在启动 train_qwen_grpo.py... (配置读取自 config.yaml)${NC}"
    $PYTHON_EXE trainer/train_qwen_grpo.py $EXTRA_ARGS
}

# 5. 运行模型评估
run_evaluate() {
    echo -e "${CYAN}[Evaluate] 正在启动评估测试脚本 scripts/eval_gee_model.py...${NC}"
    if [ -f "scripts/eval_gee_model.py" ]; then
        $PYTHON_EXE scripts/eval_gee_model.py
    else
        echo -e "${RED}[Error] 未找到 scripts/eval_gee_model.py 文件！${NC}"
    fi
}

# 6. 启动 API 服务
run_api_server() {
    # 从 config.yaml 中读取 enable_api_server 变量，校验是否允许启动
    API_ENABLED=$($PYTHON_EXE -c "import yaml, os; cfg=yaml.safe_load(open('config.yaml')) if os.path.exists('config.yaml') else {}; print(cfg.get('server', {}).get('enable_api_server', True))" 2>/dev/null)
    if [ "$API_ENABLED" = "False" ] || [ "$API_ENABLED" = "false" ]; then
        echo -e "${RED}[Forbidden] API Server 在 config.yaml 配置文件中已被禁用！${NC}"
        echo -e "若需启动，请先将 config.yaml 中 'enable_api_server' 字段修改为 'true'。"
        return 1
    fi

    echo -e "${CYAN}[API Server] 部署模型选择子菜单：${NC}"
    echo "  1) Base 原始基座模型"
    echo "  2) SFT 微调模型 (out/qwen_sft_merged)"
    echo "  3) GRPO 强化对齐模型 (out/qwen_grpo_merged)"
    echo "  4) 返回主菜单"
    read -p "请选择要加载部署的模型版本 [1-4]: " deploy_choice
    
    MODEL_PATH=""
    case $deploy_choice in
        1)
            MODEL_PATH="pretrained_models/Qwen/Qwen2___5-Coder-1___5B-Instruct"
            ;;
        2)
            MODEL_PATH="out/qwen_sft_merged"
            ;;
        3)
            MODEL_PATH="out/qwen_grpo_merged"
            ;;
        4)
            return 0
            ;;
        *)
            echo -e "${RED}无效选择，取消部署。${NC}"
            return 1
            ;;
    esac

    if [ ! -d "$MODEL_PATH" ]; then
        echo -e "${RED}[Error] 目标模型目录不存在: $MODEL_PATH。请确认是否完成了前置的微调与合并权重步骤。${NC}"
        return 1
    fi

    echo -e "${GREEN}[API Server] 正在启动 OpenAI 兼容接口服务...${NC}"
    echo -e "接口地址: ${YELLOW}http://127.0.0.1:8998/v1/chat/completions${NC}"
    $PYTHON_EXE scripts/serve_openai_api.py --load_from "$MODEL_PATH"
}

# 7. 启动 Web Demo (Streamlit)
run_web_ui() {
    # 从 config.yaml 中读取 enable_web_ui 变量，校验是否允许启动
    UI_ENABLED=$($PYTHON_EXE -c "import yaml, os; cfg=yaml.safe_load(open('config.yaml')) if os.path.exists('config.yaml') else {}; print(cfg.get('server', {}).get('enable_web_ui', True))" 2>/dev/null)
    if [ "$UI_ENABLED" = "False" ] || [ "$UI_ENABLED" = "false" ]; then
        echo -e "${RED}[Forbidden] Web UI 在 config.yaml 配置文件中已被禁用！${NC}"
        echo -e "若需启动，请先将 config.yaml 中 'enable_web_ui' 字段修改为 'true'。"
        return 1
    fi

    echo -e "${CYAN}[Web UI] Streamlit Web 网页演示程序启动...${NC}"
    
    # Streamlit scripts/web_demo.py 需要动态扫描自己同目录下的子目录以发现模型。
    # 我们自动检查 out/ 目录下的合并模型并将其软链接到 scripts/ 目录下，方便 Web UI 扫描到。
    echo -e "正在同步配置 Web UI 动态模型软链接..."
    mkdir -p out
    
    # 软链接 base 模型
    if [ -d "pretrained_models/Qwen/Qwen2___5-Coder-1___5B-Instruct" ]; then
        ln -sfn ../pretrained_models/Qwen/Qwen2___5-Coder-1___5B-Instruct scripts/Qwen2.5-Coder-1.5B-Instruct
    fi
    # 软链接 SFT 模型
    if [ -d "out/qwen_sft_merged" ]; then
        ln -sfn ../out/qwen_sft_merged scripts/Qwen2.5-Coder-SFT-Merged
    fi
    # 软链接 GRPO 模型
    if [ -d "out/qwen_grpo_merged" ]; then
        ln -sfn ../out/qwen_grpo_merged scripts/Qwen2.5-Coder-GRPO-Merged
    fi
    
    # 启动 Streamlit
    if command -v streamlit &> /dev/null; then
        echo -e "${GREEN}[Web UI] 正在启动 Streamlit，稍后浏览器将自动打开页面...${NC}"
        streamlit run scripts/web_demo.py
    else
        # 尝试通过 python 模块方式跑
        $PYTHON_EXE -m streamlit run scripts/web_demo.py 2>/dev/null
        if [ $? -ne 0 ]; then
            echo -e "${RED}[Error] 系统未安装 streamlit！请运行: pip install streamlit${NC}"
        fi
    fi
}

# 主控制循环
while true; do
    show_banner
    check_environment
    
    echo -e "${BOLD}请选择您需要执行的操作:${NC}"
    echo -e "  ${PURPLE}0)${NC} ${BOLD}[Run All] 一键全流程自动化运行 (数据准备 -> SFT -> SFT合并 -> GRPO -> GRPO合并 -> 评估)${NC}"
    echo -e "  ${GREEN}1)${NC} [Data Prep] 一键完成数据转换与划分"
    echo -e "  ${GREEN}2)${NC} [SFT Train] 启动 SFT LoRA 微调训练"
    echo -e "  ${GREEN}3)${NC} [Merge Weights] 合并 LoRA 权重 (SFT / GRPO)"
    echo -e "  ${GREEN}4)${NC} [GRPO Train] 启动 GRPO 强化对齐训练"
    echo -e "  ${GREEN}5)${NC} [Evaluate] 运行三阶段模型对比评估"
    echo -e "  ${GREEN}6)${NC} [API Server] 启动 OpenAI 兼容 API 服务"
    echo -e "  ${GREEN}7)${NC} [Web UI] 启动 Web Demo 网页演示界面"
    echo -e "  ${RED}8)${NC} 退出"
    echo -e "${BLUE}======================================================================${NC}"
    read -p "请输入对应的操作编号 [0-8]: " main_choice
    
    case $main_choice in
        0)
            run_all_pipeline
            ;;
        1)
            run_data_prep
            ;;
        2)
            run_sft_train
            ;;
        3)
            run_merge_weights
            ;;
        4)
            run_grpo_train
            ;;
        5)
            run_evaluate
            ;;
        6)
            run_api_server
            ;;
        7)
            run_web_ui
            ;;
        8)
            echo -e "${YELLOW}感谢使用，再见！${NC}"
            exit 0
            ;;
        *)
            echo -e "${RED}输入有误，请重新选择！${NC}"
            ;;
    esac
    
    echo -e "\n${YELLOW}按任意键返回主菜单...${NC}"
    read -n 1 -s
done

