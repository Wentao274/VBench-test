# VBench Local Evaluation Pipeline

使用本地部署的视频生成模型（通过 Video Gateway API）生成视频，并使用 VBench 框架进行自动化评测，最终得出 Quality / Semantic / Total 评测分数。

## 架构概览

```
run_vbench_main.py (主入口)
  │
  ├── Step 1: sample_videos.py      调用模型 API 生成视频
  │     └── Video Gateway API (POST /v1/videos)
  │
  ├── Step 2: evaluate_videos.py     使用 VBench 评测视频
  │     ├── static_filter.py          (temporal_flickering 专用过滤)
  │     └── evaluate.py               (VBench 原始评测脚本，不修改)
  │
  └── Step 3: 输出评测分数             Quality / Semantic / Total Score
```

### 新增脚本说明

| 脚本 | 说明 | 是否修改原始代码 |
|------|------|:---:|
| `sample_videos.py` | 采样桥接脚本：读取 VBench Prompt，调用模型 API 生成视频 | 否 |
| `evaluate_videos.py` | 评测脚本：调用原始 `evaluate.py` 逐维度评测，计算总分 | 否 |
| `run_vbench_main.py` | 主入口：串联 Step 1 → Step 2 → Step 3，含环境预检 | 否 |
| `compare_runs.py` | 对比工具：汇总多次运行的 `final_scores.json` 生成对比表 | 否 |

> 以上三个脚本均为独立新增文件，**不修改 VBench 仓库的任何原始源代码**。

---

## 前置条件

### 1. Python 环境

> 建议使用隔离环境（venv / conda / uv 均可）：`requirements.txt` 中有老版本 pin（`numpy<2.0.0`、`transformers==4.33.2`），直接装进系统 Python 可能与其他项目冲突。以下两种方式二选一。

**方式 A：pip（venv）**

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

**方式 B：uv**

```bash
uv venv --python 3.10
source .venv/bin/activate        # Windows: .venv\Scripts\activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
uv pip install -r requirements.txt
```

> 之后每次运行脚本前先激活环境。注意 `torch` 必须先于 `requirements.txt` 单独安装（带 CUDA 版本，requirements.txt 中未包含）。

### 2. VBench 预训练模型权重

VBench 评测依赖多个预训练模型（DINO、CLIP、GRiT、RAFT、UMT、ViCLIP 等）。首次使用前需下载到 `~/.cache/vbench/`。

**方式一**：按照 `pretrained/` 目录下各子文件夹中的 `model_path.txt` 指引手动下载。

**方式二**：首次运行评测时，`evaluate.py` 会自动通过 `wget` 下载（需要系统有 `wget` 命令）。

**方式三**（推荐 Windows 用户）：手动下载后使用 `--load_ckpt_from_local` 标志：

```bash
# 下载模型到 ~/.cache/vbench/ 后
python run_vbench_main.py ... --load_ckpt_from_local
```

### 3. detectron2（部分维度需要）

`object_class`、`multiple_objects`、`color`、`spatial_relationship` 四个维度依赖 detectron2：

```bash
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git
# uv 用户等价：uv pip install detectron2@git+https://github.com/facebookresearch/detectron2.git
```

> 如果不需要这四个维度，可跳过安装，评测时会自动标记为失败，不影响其他维度。

### 4. Video Gateway 服务

确保本地 Video Gateway 服务已启动：

```bash
curl -sS http://localhost:8000/v1/models | jq
```

### 5. GPU

VBench 评测需要 CUDA GPU（`evaluate.py` 中 `device = torch.device("cuda")`）。

---

## 在 GPU 机器上部署（推荐架构）

流水线跑在一台有 CUDA 的机器上，通过 HTTP 访问被测模型服务（MiniMax-H3 + Video Gateway）。模型与评测机**是否同机均可**，区别只是网关地址不同：

```
被测模型所在机器                         评测机（GPU，执行本流水线）
MiniMax-H3 + Video Gateway   ←──HTTP──  run_vbench_main.py（主入口）
                                             ├─ sample_videos.py   提交/轮询/下载，视频落盘
                                             └─ evaluate_videos.py  本地 CUDA 评测，出分
```

> **流程逻辑**：桥接脚本读取 VBench 官方提示词 → 调用模型 API 生成视频 → 生成的视频作为输入，交给原版 VBench 评测。新增脚本只做流程编排和环境准备，不改变 VBench 原始评测逻辑。

下文 `<GATEWAY_URL>` 指你的网关地址，按实际部署替换：
- 模型与评测**同机**：`http://localhost:8000`
- 模型在**其他机器**：`http://<IP>/<路由名>`，如 `http://10.xxx.xxx.xxx/sp-xxx-xxxx`（支持带路径前缀的网关；真实地址填在 `vbench.env`，不入库）

### 首次部署：从零到出分（逐步执行）

#### 第 1 步：获取代码

```bash
git clone https://github.com/Wentao274/VBench-test.git
cd VBench-test
```

#### 第 2 步：安装 Python 环境（每台机器一次）

**做什么**：安装三类依赖。PyTorch 是 GPU 版且必须**先单独装**（requirements.txt 里没有）；detectron2 只有 `object_class` / `multiple_objects` / `color` / `spatial_relationship` 4 个维度需要，不测可跳过。

```bash
# venv 用户先执行: python -m venv .venv && source .venv/bin/activate
# uv   用户先执行: uv venv --python 3.10 && source .venv/bin/activate，并把下面的 pip 换成 uv pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git   # 可选
```

**验收**：

```bash
python -c "import torch; print(torch.cuda.is_available())"   # 期望输出 True
```

#### 第 3 步：预下载评测权重（每台机器一次）

**做什么**：16 个维度的评测要用十几个第三方模型文件（DINO、CLIP、GRiT、RAFT、UMT、ViCLIP 等，合计数 GB）。提前调用**原版** `vbench.utils.init_submodules` 下到 `~/.cache/vbench/`，避免评测中途才发现缺权重。

```bash
python evaluate_videos.py --all --download_weights
```

**验收**：输出 `Weight download complete`；`ls ~/.cache/vbench/` 出现 `amt_model`、`raft_model`、`grit_model` 等目录。

> 只测部分维度时把 `--all` 换成 `--dimensions ...`，只下载所选维度所需权重。

#### 第 4 步：环境体检

**做什么**：正式开跑前一次性检查 6 项，全过再继续。正式运行时这些检查也会自动先执行。

```bash
python run_vbench_main.py \
    --gateway_url <GATEWAY_URL> \
    --model minimax-h3 --all \
    --output_dir ./sampled_videos/h3 \
    --preflight
```

| 检查项 | 查什么 | 不通过怎么办 |
|---|---|---|
| Gateway | 网关可达、`minimax-h3` 在模型列表 | 核对地址与网络连通性 |
| 磁盘 | 按 6220 条视频估算 ~95GB | 清理空间或换目录 |
| CUDA | torch 可用、各卡显存余量（<8GB 告警） | 重装 CUDA 版 torch；显存紧张用 `--gpu_id` 换卡 |
| 权重 | `~/.cache/vbench/` 按维度逐一核对 | 回到第 3 步补齐 |
| detectron2 | 选了 GRiT 4 维度时才检查 | 回到第 2 步，或去掉这 4 个维度 |
| 视频 | `--skip_sample` 模式下目录里有视频 | `--output_dir` 指向正确的目录 |

**验收**：`0 preflight check(s) FAILED`。

#### 第 5 步：冒烟测试（约 15 分钟）

**做什么**：只取 1 个提示词生成 5 条视频，走完「生成 → 下载 → 落盘 → 评测 → 出分」全链路。全通了再跑全量，避免全量跑几小时后在最后一环翻车。

```bash
python run_vbench_main.py \
    --gateway_url <GATEWAY_URL> \
    --model minimax-h3 \
    --dimensions subject_consistency \
    --max_prompts 1 \
    --output_dir ./sampled_videos/smoke
```

**验收**：
- 采样统计 `success 5 / failed 0`；若报 `submit HTTP 400` 且提示 `generate_sound`，说明该网关版本不认识此字段（需改一行代码，见 FAQ）
- `sampled_videos/smoke/<时间戳>/` 下有 5 个 `.mp4`，手动播放确认画面正常
- 终端分数表 `subject_consistency` 有 [0,1] 区间分数，`evaluation_results/final_scores.json` 已生成

> 首次评测 `subject_consistency` 会联网下载 DINO 权重，多等几分钟属正常。

#### 第 6 步：正式全量评测（数小时，建议 tmux）

**做什么**：16 维度 × 944 提示词 = 6220 条视频 + 全量评测。在 `tmux` / `screen` 中运行，SSH 断线不影响：

```bash
tmux new -s vbench
python run_vbench_main.py \
    --gateway_url <GATEWAY_URL> \
    --model minimax-h3 --all \
    --output_dir ./sampled_videos/h3
# 按 Ctrl+B 再按 D 脱离会话；重新登录后 tmux attach -t vbench 回来看进度
```

中断（断网/断电/手动 Ctrl+C）后**重跑同一命令**即可续跑：已落盘的视频自动跳过，已成功的评测维度自动复用，从断点继续。

> 已按下文「配置文件运行」创建 `vbench.env` 的话，本步命令可简化为 `./run.sh`。

#### 第 7 步：查看与对比结果

**结果位置**：`<output_dir>/<时间戳>/evaluation_results/final_scores.json`，运行结束时终端也会打印 16 维度分数表 + Quality / Semantic / Total 汇总。

多次评测（不同模型/参数）后一键对比：

```bash
python compare_runs.py \
    run1/evaluation_results/final_scores.json \
    run2/evaluation_results/final_scores.json
```

### 长任务与多卡提示

- 与模型 serving 共享 GPU 时注意显存：预检会报告各卡余量，评测约需 8GB 以上空闲
- 多卡节点用 `--gpu_id 1` 指定空闲卡（等价 `CUDA_VISIBLE_DEVICES=1`）
- Linux 缺权重时评测过程会用 `wget` 自动下载（建议第 3 步提前拉齐）
- 更多命令变体（dry_run 预览、只采样、只评测、指定维度等）见下文「快速开始」

---

## 配置文件运行（推荐：免敲重复参数）

`gateway_url`、`model`、`output_dir` 这类每次都一样的参数集中到 `vbench.env`，日常只敲 `./run.sh`；临时参数在命令行追加即可（命令行优先于配置文件）。

### 一次性设置（Linux 评测机，约 1 分钟）

```bash
cp vbench.env.example vbench.env   # 1. 从模板创建真实配置
vim vbench.env                     # 2. 只需替换 GATEWAY_URL 等真实值
chmod +x run.sh                    # 3. 赋予执行权限（之后 ./run.sh 直接可用）
```

`vbench.env` 配置项说明：

| 配置项 | 必填 | 说明 |
|---|---|---|
| `GATEWAY_URL` | 是 | 网关地址。模型同机：`http://localhost:8000`；模型在其他机器：`http://<IP>/<路由名>` |
| `MODEL` | 是 | 模型名，必须出现在网关 `GET /v1/models` 列表中 |
| `OUTPUT_DIR` | 是 | 采样输出根目录，每次运行自动创建时间戳子目录 |
| `GPU_ID` | 否 | 多卡节点指定空闲卡（留空 = 默认 0 号卡） |
| `DIMENSIONS` | 否 | 只测部分维度，空格分隔（留空 = 全部 16 个维度） |
| `LOAD_LOCAL` | 否 | `true` = 附加 `--load_ckpt_from_local`（内网受限机器用，下载权重时也要同样设置） |
| `API_TOKEN` | 否 | 预留。将来网关增加鉴权时填，当前脚本尚未使用 |

### 日常使用

```bash
./run.sh                                                  # 按配置全量运行（建议 tmux 中）
./run.sh --preflight                                      # 环境体检
./run.sh --skip_sample                                    # 只评测最新一批已生成的视频
./run.sh --dimensions subject_consistency --max_prompts 1 # 冒烟测试（临时参数覆盖配置）
```

> **覆盖规则**：配置文件管每次都一样的参数，命令行管本次临时参数；两者冲突时命令行优先。
> `run.sh` 每次运行会先打印实际执行的完整命令，便于核对参数。

### 敏感信息管理

- **真实配置 `vbench.env` 不入库**：已加入 `.gitignore`，内网地址、未来的鉴权 token 等敏感值只存在于此文件
- **入库的是模板 `vbench.env.example`**：只含占位符，无真实值
- 将来 Gateway 增加鉴权时，在 `vbench.env` 中填 `API_TOKEN` 即可，同样不会提交

---

## 快速开始

### 一键运行（生成 + 评测 + 出分）

```bash
python run_vbench_main.py \
    --gateway_url http://localhost:8000 \
    --model minimax-h3 \
    --all \
    --output_dir ./sampled_videos/my_model \
    --load_ckpt_from_local
```

此命令会：
1. 运行环境预检（Gateway 可达 / CUDA / 磁盘空间）
2. 在 `--output_dir` 下创建时间戳子目录 `YYYYMMDD_HHMMSS`
3. 读取全部 16 个维度的 VBench Prompt（944 个唯一提示词）
4. 调用模型 API 生成 6220 个视频，保存到时间戳目录
5. 逐维度运行 VBench 评测
6. 输出 Quality / Semantic / Total 分数

> 每次运行会在 `--output_dir` 下创建独立的时间戳子目录，多次测试互不覆盖。
> 若同一秒内启动两次，自动追加 `_N` 后缀避免冲突。

### 环境预检（不启动流水线）

```bash
python run_vbench_main.py \
    --gateway_url http://localhost:8000 \
    --model minimax-h3 --all \
    --output_dir ./sampled_videos/my_model \
    --preflight
```

检查项：Gateway 可达性、磁盘空间（按视频数估算）、CUDA 可用性、**各卡显存余量**（与 serving 共存时重点看）、**评测权重就绪情况**（`~/.cache/vbench/` 按维度逐一核对）、detectron2（选了 GRiT 维度时）。全部通过后退出。
正式运行时预检也会**自动执行**，关键项失败则直接中止，避免跑了几小时采样后才发现环境问题。

### 烟雾测试（先小规模验证端到端流程）

首次使用建议先只取 1 个提示词生成 5 条视频，快速验证「生成 → 评测 → 出分」全链路（与部署指南第 5 步相同）：

```bash
python run_vbench_main.py \
    --gateway_url http://localhost:8000 \
    --model minimax-h3 \
    --dimensions subject_consistency \
    --max_prompts 1 \
    --output_dir ./sampled_videos/smoke \
    --load_ckpt_from_local
```

### 预览执行计划（不实际调用 API）

```bash
python run_vbench_main.py \
    --gateway_url http://localhost:8000 \
    --model minimax-h3 --all \
    --output_dir ./sampled_videos/my_model \
    --dry_run
```

### 只评测部分维度

```bash
python run_vbench_main.py \
    --gateway_url http://localhost:8000 \
    --model minimax-h3 \
    --dimensions subject_consistency color overall_consistency \
    --output_dir ./sampled_videos/my_model
```

### 只评测已有视频（跳过采样）

跳过 Step 1，直接对已生成的视频进行评测。自动使用 `--output_dir` 下**最新**的时间戳目录：

```bash
python run_vbench_main.py \
    --gateway_url http://localhost:8000 \
    --model minimax-h3 --all \
    --output_dir ./sampled_videos/my_model \
    --skip_sample
```

---

## 分步执行

### Step 1: 生成视频

> 分步执行时，需要手动指定时间戳目录。推荐直接使用 `run_vbench_main.py`，它会自动创建时间戳目录并串联两步。

```bash
python sample_videos.py \
    --gateway_url http://localhost:8000 \
    --model minimax-h3 \
    --all \
    --output_dir ./sampled_videos/my_model/20260923_153045 \
    --concurrency 4 \
    --seconds 5 \
    --size 1344x768 \
    --num_inference_steps 50
```

**关键参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--gateway_url` | (必填) | Video Gateway 地址 |
| `--model` | `minimax-h3` | 模型名称 |
| `--all` | - | 生成全部 16 个维度的视频 |
| `--dimensions` | - | 指定维度，与 `--all` 互斥 |
| `--output_dir` | (必填) | 视频输出目录 |
| `--seconds` | 5 | 视频时长（4-15 秒） |
| `--size` | `1344x768` | 视频尺寸 |
| `--num_inference_steps` | 50 | 推理步数 |
| `--concurrency` | 4 | 并发生成数 |
| `--base_seed` | 0 | 基础随机种子 |
| `--max_prompts` | 0 | 限制提示词数量（烟雾测试），0 = 不限制 |
| `--generate_sound` | 关 | 同时生成音频（VBench 不评音频，默认关闭以节省算力） |
| `--max_fail_rate` | 0.05 | 失败率超过此值时以错误码退出 |
| `--dry_run` | - | 仅预览计划 |

**特性：**
- 断点续采：已存在的视频自动跳过，中断后重跑同一命令即可继续
- 幂等提交：每个 (prompt, index) 生成唯一 `Idempotency-Key`，重试不会创建重复任务
- 确定性种子：基于 `base_seed + md5(prompt-index)` 生成可复现的随机种子
- 失败率拦截：失败率超过 `--max_fail_rate`（默认 5%）时以错误码退出，主入口据此中止流水线，避免评测不完整数据
- 超时任务复查：轮询超时被标记为失败的任务，会在结束时向服务端做一次最终状态复查，抢救"迟到"的完成结果
- 采样日志：生成 `{output_dir}/_sampling_log.json` 记录所有任务状态

### Step 2: 评测视频

```bash
python evaluate_videos.py \
    --videos_path ./sampled_videos/my_model/20260923_153045 \
    --all \
    --output_path ./sampled_videos/my_model/20260923_153045/evaluation_results \
    --load_ckpt_from_local
```

**关键参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--videos_path` | (必填) | 视频目录（Step 1 的 output_dir） |
| `--all` | - | 评测全部 16 个维度 |
| `--dimensions` | - | 指定维度 |
| `--output_path` | `./evaluation_results` | 评测结果输出目录 |
| `--load_ckpt_from_local` | - | 使用本地下载的模型权重 |
| `--skip_static_filter` | - | 跳过 temporal_flickering 的静态过滤 |
| `--force` | - | 强制重评已有成功结果的维度（默认复用跳过） |
| `--skip_integrity_check` | - | 跳过评测前的视频完整性校验 |
| `--download_weights` | - | 预下载所选维度的评测权重后退出（不需视频） |
| `--ngpus` | 1 | GPU 数量 |
| `--keep_temp` | - | 保留临时目录 |

**评测流程：**
1. **完整性校验**：对照 Prompt 套件检查视频数量缺口（WARNING）；检测零字节/损坏视频并自动排除（避免评测崩溃）
2. **结果合并**：读取已有 `final_scores.json`，本次已成功的维度直接复用（`--force` 可强制重评）
3. 创建临时目录，通过硬链接只包含 `.mp4` 文件（解决 VBench `os.listdir` 取后缀的问题）
4. 对 `temporal_flickering` 维度，先运行 `static_filter.py` 过滤静态视频
5. 逐维度调用原始 `evaluate.py` 进行评测
6. **合并写出**：本次结果与历史成功结果合并后写入 `final_scores.json`（旧文件备份为 `.bak`），重评部分维度**不会丢失**其他维度的结果
7. 计算 Quality / Semantic / Total Score

> 归一化与官方 `scripts/cal_final_score.py` 完全一致（不做 [0,1] 截断）；评测失败的维度不参与计分，仅在结果表中标记 FAIL。

### Step 3: 查看分数

评测完成后，分数保存在 `{eval_output_path}/final_scores.json`，同时终端会打印结果表格：

```
===================================================================
                    VBench Evaluation Results
===================================================================
  Dimension                        Raw    Normalized  Status
-----------------------------------------------------------------
  subject_consistency            0.9512       0.9438
  background_consistency         0.8734       0.8265
  temporal_flickering            0.9821       0.9522
  ...
-----------------------------------------------------------------
  Quality Score                    0.8123
  Semantic Score                    0.7456
  Total Score                      0.7990
===================================================================
```

### 对比多次运行的分数

多次运行（不同模型版本 / 参数 / 时间戳目录）后，可用 `compare_runs.py` 生成对比表：

```bash
# 方式一：指定多个运行目录
python compare_runs.py \
    ./sampled_videos/my_model/20260923_153045 \
    ./sampled_videos/my_model/20260924_091030

# 方式二：自动发现基础目录下所有含评测结果的时间戳目录
python compare_runs.py --base ./sampled_videos/my_model
```

每行标注各次运行中的最优值（`*`），未评测的维度显示 `-`。

---

## 完整参数列表（run_vbench_main.py）

```bash
python run_vbench_main.py \
    --gateway_url http://localhost:8000 \      # Video Gateway 地址
    --model minimax-h3 \                       # 模型名称
    --all \                                    # 全部 16 维度
    --output_dir ./sampled_videos/my_model \   # 基础目录（时间戳子目录会创建在此下）
    # --- 采样参数 ---
    --seconds 5 \                              # 视频时长
    --size 1344x768 \                          # 视频尺寸
    --fps 24 \                                 # 帧率
    --num_inference_steps 50 \                 # 推理步数
    --base_seed 0 \                            # 基础种子
    --concurrency 4 \                          # 并发数
    --poll_interval 10 \                       # 轮询间隔（秒）
    --poll_timeout 900 \                       # 单任务超时（秒）
    --max_prompts 0 \                          # 限制提示词数（烟雾测试）
    --generate_sound \                         # 生成音频（默认关闭）
    --max_fail_rate 0.05 \                     # 采样失败率阈值
    # --- 评测参数 ---
    --eval_output_path ./eval_results \        # 评测结果目录（默认 {run_dir}/evaluation_results）
    --load_ckpt_from_local \                   # 使用本地模型权重
    --skip_static_filter \                     # 跳过静态过滤
    --ngpus 1 \                                # GPU 数量
    --gpu_id 1 \                               # 指定空闲卡（等价 CUDA_VISIBLE_DEVICES）
    --force \                                  # 强制重评已有结果的维度
    --skip_integrity_check \                   # 跳过完整性校验
    # --- 流程控制 ---
    --skip_sample \                            # 跳过采样（评测已有视频）
    --skip_eval \                              # 跳过评测（只生成视频）
    --dry_run \                                # 预览不执行
    --preflight                                # 仅运行环境预检
```

---

## 评测维度说明

VBench 共 16 个评测维度，分为两大类：

### Quality Score（技术质量，权重 4）

| 维度 | 说明 | 每提示词视频数 |
|------|------|:---:|
| `subject_consistency` | 主体一致性（DINO） | 5 |
| `background_consistency` | 背景一致性（CLIP） | 5 |
| `temporal_flickering` | 时序闪烁 | 25 |
| `motion_smoothness` | 运动平滑度（AMT） | 5 |
| `aesthetic_quality` | 美学质量（CLIP+LAION） | 5 |
| `imaging_quality` | 成像质量（MUSIQ） | 5 |
| `dynamic_degree` | 动态程度（RAFT） | 5 |

### Semantic Score（语义遵循，权重 1）

| 维度 | 说明 | 每提示词视频数 |
|------|------|:---:|
| `object_class` | 物体类别（GRiT） | 5 |
| `multiple_objects` | 多物体（GRiT） | 5 |
| `human_action` | 人体动作（UMT） | 5 |
| `color` | 颜色（GRiT） | 5 |
| `spatial_relationship` | 空间关系（GRiT） | 5 |
| `scene` | 场景识别（Tag2Text） | 5 |
| `appearance_style` | 外观风格（CLIP） | 5 |
| `temporal_style` | 时序风格（ViCLIP） | 5 |
| `overall_consistency` | 整体一致性（ViCLIP） | 5 |

### 总分计算

```
Total Score = (Quality Score * 4 + Semantic Score * 1) / 5
```

每个维度分数经过归一化：`(score - min) / (max - min)`，归一化参数见 `scripts/constant.py`。

---

## 文件结构

运行后产生的文件结构：

```
VBench-test/
├── sample_videos.py              # [新增] 采样桥接脚本
├── evaluate_videos.py            # [新增] 评测脚本
├── run_vbench_main.py            # [新增] 主入口
├── compare_runs.py               # [新增] 多次运行分数对比
├── README_LOCAL.md               # [新增] 本文档
│
├── evaluate.py                   # [原始] VBench 评测入口
├── static_filter.py              # [原始] 静态视频过滤
├── vbench/                       # [原始] VBench 核心库
├── prompts/                      # [原始] Prompt 套件
│
└── sampled_videos/my_model/      # --output_dir 基础目录
    ├── 20260923_153045/          # 第 1 次运行（时间戳子目录）
    │   ├── a person swimming in ocean-0.mp4
    │   ├── a person swimming in ocean-1.mp4
    │   ├── ...
    │   ├── _sampling_log.json     # 采样日志
    │   └── evaluation_results/   # 评测结果
    │       ├── subject_consistency/
    │       │   └── results_*_eval_results.json
    │       ├── color/
    │       │   └── results_*_eval_results.json
    │       ├── static_filter_results/  # temporal_flickering 过滤结果
    │       │   └── filtered_videos/
    │       └── final_scores.json     # 最终分数汇总
    │
    └── 20260924_091030/          # 第 2 次运行（互不覆盖）
        ├── a person swimming in ocean-0.mp4
        ├── ...
        └── evaluation_results/
            └── final_scores.json
```

> 每次运行自动创建 `YYYYMMDD_HHMMSS` 时间戳子目录，多次测试数据互不覆盖。使用 `--skip_sample` 时自动评测最新目录。

---

## 常见问题

### Q: 中断后如何继续？

**采样中断**：直接重跑同一命令。`sample_videos.py` 会自动跳过已存在的视频。

**评测中断**：直接重跑同一命令。已成功的维度自动复用（结果表中标记 `(reused)`），只重评未完成/失败的维度。

### Q: 某些维度评测失败怎么办？

评测失败不会影响其他维度，失败的维度不参与计分（仅标记 FAIL）。查看 `final_scores.json` 中 `details` 字段的 `status`，失败的维度标记为 `eval_failed`。常见原因：
- 模型权重未下载 → 使用 `--load_ckpt_from_local` 或手动下载
- detectron2 未安装 → 安装 detectron2 或跳过相关维度
- GPU 显存不足 → 减小 `--ngpus` 或单独评测失败的维度

### Q: 重新评测会覆盖已有结果吗？

不会。`evaluate_videos.py` 采用**合并写入**：
- 本次请求的维度中已有成功结果的 → 自动复用跳过（`--force` 强制重评）
- 本次未请求的维度 → 保留历史结果合并进新的 `final_scores.json`
- 覆盖前旧文件自动备份为 `final_scores.json.bak`

### Q: 如何只重新评测失败的维度？

```bash
python evaluate_videos.py \
    --videos_path ./sampled_videos/my_model/20260923_153045 \
    --dimensions object_class multiple_objects \
    --output_path ./sampled_videos/my_model/20260923_153045/evaluation_results \
    --force \
    --load_ckpt_from_local
```

> 注意 `--force`：不加则已有成功结果的维度会被复用跳过；加了则指定维度全部重评，其余维度的历史结果仍会合并保留。

### Q: Windows 上 `wget` 不可用怎么办？

手动下载模型权重到 `~/.cache/vbench/`，然后使用 `--load_ckpt_from_local` 标志。各模型下载地址见 `pretrained/` 目录下的 `model_path.txt` 文件。

### Q: 采样失败率过高会怎样？

`sample_videos.py` 在失败率超过 `--max_fail_rate`（默认 5%）时以错误码退出，`run_vbench_main.py` 据此中止流水线，避免对残缺数据出分。失败率低于阈值时仅提示，可重跑同一命令补齐（已生成视频自动跳过）。

### Q: 如何对比多次运行 / 多个模型的分数？

```bash
python compare_runs.py --base ./sampled_videos/my_model
```

### Q: 磁盘空间不够怎么办？

全部 16 维度约需 100-200 GB。多次运行的时间戳目录会累积占用，可删除旧的 `{timestamp}` 目录（或先备份其中的 `evaluation_results/final_scores.json` 供 `compare_runs.py` 对比）。运行主入口时的预检也会给出磁盘空间估算。

### Q: 评测需要多长时间？

- 采样：取决于模型推理速度和并发数，6220 个视频通常需要数小时
- 评测：取决于 GPU 和维度，单维度通常 1-10 分钟，全部 16 维度约 30-60 分钟

### Q: 视频文件名中的特殊字符会出问题吗？

VBench Prompt 套件已检查过，不含 Windows 非法字符（`< > : " \ | ? *`）。文件名格式为 `{prompt}-{index}.mp4`，与 VBench `vbench_standard` 模式完全匹配。

### Q: 为什么需要创建临时目录？

VBench 的 `build_full_info_json` 方法通过 `os.listdir(videos_path)[0]` 获取文件后缀。如果目录中有非 `.mp4` 文件（如 `_sampling_log.json`）排在前面，后缀会错误地变为 `.json`，导致找不到视频。`evaluate_videos.py` 创建只含 `.mp4` 硬链接的临时目录来规避此问题，不修改原始代码。
