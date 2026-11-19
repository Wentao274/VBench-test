#!/usr/bin/env bash
# ============================================================
# VBench 流水线包装脚本
#
# 固定参数从 vbench.env 读取；命令行参数追加在后，可临时覆盖配置。
#
# 用法:
#   ./run.sh                                          按 vbench.env 全量运行
#   ./run.sh --preflight                              环境体检
#   ./run.sh --skip_sample                            只评测最新一批已生成的视频
#   ./run.sh --dimensions subject_consistency --max_prompts 1   冒烟测试
#
# 规则: 配置文件管每次都一样的参数，命令行管本次临时参数；
#       两者冲突时命令行优先。
# ============================================================
set -euo pipefail

ENV_FILE="$(cd "$(dirname "$0")" && pwd)/vbench.env"
if [ ! -f "$ENV_FILE" ]; then
    echo "ERROR: vbench.env not found." >&2
    echo "       Run: cp vbench.env.example vbench.env  then fill in real values." >&2
    exit 1
fi
source "$ENV_FILE"

: "${GATEWAY_URL:?vbench.env: GATEWAY_URL not set}"
: "${MODEL:?vbench.env: MODEL not set}"
: "${OUTPUT_DIR:?vbench.env: OUTPUT_DIR not set}"

ARGS=(--gateway_url "$GATEWAY_URL" --model "$MODEL" --output_dir "$OUTPUT_DIR")

if [ -n "${GPU_ID:-}" ]; then
    ARGS+=(--gpu_id "$GPU_ID")
fi
if [ "${LOAD_LOCAL:-}" = "true" ]; then
    ARGS+=(--load_ckpt_from_local)
fi

# --all 在脚本内部的优先级高于 --dimensions，因此若命令行带了
# --dimensions，则不能再附加 --all，否则命令行维度会被 --all 遮蔽。
CLI_HAS_DIMS=false
for a in "$@"; do
    case "$a" in
        --dimensions|--dimension|-d|--dimensions=*) CLI_HAS_DIMS=true ;;
    esac
done

if [ -n "${DIMENSIONS:-}" ]; then
    # 空格分隔的多个维度，需要拆分为多个参数
    ARGS+=(--dimensions $DIMENSIONS)
elif [ "$CLI_HAS_DIMS" = false ]; then
    ARGS+=(--all)
fi

echo "run.sh: python run_vbench_main.py ${ARGS[*]} $*"
python run_vbench_main.py "${ARGS[@]}" "$@"
