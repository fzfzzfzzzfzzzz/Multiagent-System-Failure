# 运行手册

工作目录为项目根目录。安装：`python -m pip install -e ".[dev,remote]"`。默认模型为 Qwen3-8B；配置位于 `configs/models.yaml`，正式矩阵位于 `configs/study.yaml`。

## 服务器

2026-09-28 已实测 SSH：151 直连；158 经文档的 157 跳板连接。脚本只从环境变量或指定 PDF 临时读取密码，不将凭据写入源码。

```powershell
python scripts/remote.py inspect --host 151 --credentials-pdf fangc-README.pdf
python scripts/remote.py inspect --host 158 --credentials-pdf fangc-README.pdf
```

若原环境没有 gpustat，项目已经在 151 的 `/data/fangc/teamlearn/gpu_tools` 单独安装。可以使用：

```bash
PYTHONPATH=/data/fangc/teamlearn/gpu_tools /data/fangc/envs/vllm-0.8.5/bin/python -m gpustat --no-color
```

先检查空闲显存。`scripts/serve_8b_smoke.sh` 是实测可运行配置：GPU 6/7、TP2、16K 上下文、端口 18151、只监听回环地址；启动前检查两卡占用小于 512 MB。更换卡后相应修改检查列表和 CUDA_VISIBLE_DEVICES。不能停止其他项目的模型进程。

```powershell
python scripts/remote.py upload --host 151 --credentials-pdf fangc-README.pdf --local scripts/serve_8b_smoke.sh --remote /home/fangc/teamlearn/serve_8b_smoke.sh
```

在服务器的 tmux 会话中运行 `bash /home/fangc/teamlearn/serve_8b_smoke.sh`；数据、日志放 `/data/fangc/teamlearn/`，代码放 `/home/fangc/teamlearn/`。本地保持隧道运行：

```powershell
python scripts/remote.py tunnel --host 151 --credentials-pdf fangc-README.pdf --local-port 18151 --remote-port 18151
python -m teamlearn doctor --model qwen3_8b
```

14B 的模型目录为 `/data/fangc/models/Qwen3-14B`，用同一 vLLM 环境，服务名 `Qwen3-14B`，端口 18152，GPU 与 TP 数按空闲资源配置。27B 目录为 `/data/fangc/models/Qwen3.8-27B`，两节点均有；可用已有 `/data/fangc/envs/sglang-qwen38-260915` 环境，服务名需设为 `Qwen3.8-27B`、端口 18158，并暴露 tokenizer 端点。8B 已做真实推理验证；14B/27B 已核实目录及服务器可达，未启动大模型验证，doctor 需在正式运行时先通过。

只有命令实际返回对应模型名，才把该模型计入异构实验。不得通过修改 model 名称把同一个服务冒充三种模型。异构映射：协调者 27B、提供者 8B、分析者 14B、执行者 8B。所有比较方法使用同一映射。

通用启动脚本为 `scripts/serve_model.sh`，参数依次为 `8b/14b/27b`、GPU ID 列表、服务端口。在 158 首次使用时，需要在项目独立目录安装 gpustat：`python -m pip install --target /data/fangc/teamlearn/gpu_tools gpustat`（使用该节点模型环境 Python）。

27B 同时在另一个 tmux 窗格运行 CPU tokenizer 服务：

```bash
/data/fangc/envs/sglang-qwen38-260915/bin/python /home/fangc/teamlearn/tokenizer_server.py --model-path /data/fangc/models/Qwen3.8-27B --served-model-name Qwen3.8-27B --port 18159
```

将 `scripts/tokenizer_server.py` 上传到对应路径，并分别为 18158 推理和 18159 tokenizer 建 SSH 隧道。这样无需依赖 SGLang 是否原生提供兼容的 `/tokenize` 路由。客户端遇到响应模型名不匹配、缺少 usage 或 tokenizer 失败会停止，不能以估算数字继续正式对照。

## 正式研究阶段

先查看阶段，不启动实验：

```powershell
python scripts/run_required.py --plan
```

依次为数据、开发源采集、E0、训练源采集、联合接收者策略训练、开发干预、冻结、测试源采集、确认性实验、异构及 27B 同构 E7、顺序流、分析。异构阶段前需启动三种模型及对应隧道。单阶段示例：

```powershell
python scripts/run_required.py --stage collect-dev
python scripts/run_required.py --stage pilot
```

完整执行：`python scripts/run_required.py --stage all`。该命令会运行大量模型调用，不是冒烟测试。若 E0 缺少 F1–F4 任一自然源失败，会停止并写 `pilot_gate.json`；这是研究门槛失败，不是需要伪造数据填满矩阵。

E1/E3 是核心确认性对照，E2 枚举用于机制分析，E4 检查反转与角色变化，E5 比较预算曲线，E7 运行 SQLite 应用工作流，BASELINES 保留 B0–B11，E6 是可见反馈下的学习扩展。源报告只生成一次供各组复用。

单独执行可用：

```powershell
python -m teamlearn run --experiment E3 --split dev --dataset datasets --sources runs/study_v1/collected --out runs/my_e3 --model qwen3_8b
python -m teamlearn run --experiment E7 --split test --dataset datasets --sources runs/study_v1/collected --out runs/my_heterogeneous_e7 --model qwen3_8b --heterogeneous --policy runs/study_v1/policy.json --freeze runs/study_v1/frozen.json
```

真实静态实验每完成一个团队运行就保存独立 episode 和一条结果；相同配置重复命令会跳过已完成 run ID。模型/代码/数据/种子/配置变化须新建输出目录。E6 保存阶段快照但暂不支持自动恢复到中间步骤；如中断，使用新目录重跑该流，避免不完整状态续写。

## 输出与分析

每次静态运行保存真实模型输出、完整可见输入、角色工具事件、激活报告、原子事实 ID、证据哈希、首轮/最终结果、复发、API usage、tokenizer 结果、模型配置和失败日志。

```powershell
python -m teamlearn analyze --input runs/study_v1/test_E3/results.jsonl --out analysis/E3
```

输出 CSV、配对 CI、块置换和 Holm 结果、匹配交互以及含 CI 的成本图。分析工具默认拒绝 mock 数据。不要合并两个实验中重复的同一运行文件，也不要将所有种子当成独立任务。

参考规模只用于配置，不作功效保证。默认数据有 26 个独立规则家族、6 种机制、2 后端、每家族单元 8 个源候选和 6 个目标。全矩阵实际运行数取决于采集到的自然源失败数量，可用 `teamlearn plan` 查看每源的实验倍数。
