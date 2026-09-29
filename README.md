# Team Failure Learning

研究问题：同一份团队失败证据，向哪些角色提供哪些内容，能否减少新任务中的同类错误？

实现依据：[原研究方案](多智能体失败报告与跨任务学习_完整论文研究方案.md)。框架、E0 操作性检查和冻结的 Qwen3-8B H1/H4 独立确认均已完成。主结果没有支持预注册 same 总体上的 H1/H4；详见 [核心确认记录](docs/RUNNING_CORE.md)与 `artifacts/core_confirmation/REPORT.md`。

## 快速开始

```powershell
python -m pip install -e ".[dev,remote]"
python -m pytest -q
python scripts/run_required.py --plan
```

本地无模型软件验证：

```powershell
python scripts/smoke.py --model mock --out runs/my_mock_smoke
```

启动模型与 SSH 隧道后，检查服务：

```powershell
python -m teamlearn doctor --model qwen3_8b
python scripts/smoke.py --model qwen3_8b --out runs/my_live_smoke
```

正式实验按阶段执行，详细说明见 [运行手册](docs/RUNBOOK.md)。本次已按阶段门槛停止；以下命令只用于新的独立冻结运行，不应覆盖现有结果：

```powershell
python scripts/run_required.py --stage all
```

## 实现范围

| 研究计划 | 代码与产物 |
|---|---|
| F1–F6、多角色权限、局部/全局验收 | `environment.py`；工作流及实际 SQLite 事务后端 |
| 家族划分与新实例、角色变化、条件反转 | `datasets.py`；8 个训练、6 个开发、12 个测试规则家族 |
| S/O/C/D/K/P/V/U 原子证据、来源与失效 | `reports.py`；不可变哈希、状态、supersedes、角色视图 |
| B0–B11 | 无记忆、成功、规则、全历史、统一、独立反思、责任者、协调者、角色检索、G-Memory 启发适配、规则、学习分配 |
| E0–E5、E7 | `experiments.py` + `pipeline.py`；随机配对顺序、断点续跑 |
| E6 | `stream.py`；L0–L3、同库验证、只读探针与快照 |
| 统计与成本 | `analysis.py`；家族聚类 bootstrap、块置换、Holm、交互量、成本图 |
| 正式冻结与运行 | `run_required.py`；源码、模型、数据、策略、种子及推理预算冻结 |

研究代码在 [teamlearn](teamlearn/)；方法与实现差异在 [协议说明](protocol/PROTOCOL.md)；最终验证记录在 [交付状态](docs/STATUS.md)。

E8 同题恢复是原计划明确标注的可选附录项目，本次没有把它列入必须运行集合。没有将日志回放伪装成恢复实验。

## 结果解释边界

- `mock-NOT-RESEARCH` 仅检查软件，不响应报告语义，不能计算方法有效性；统计工具默认拒绝 mock 数据。
- SQLite 后端是真实读写数据库的受控应用工作流。没有下载或改编 CooperBench，不应称为 CooperBench 成绩，也不能单凭这个后端声称真实企业泛化。
- G-Memory 采用明确标注的启发式适配，保留分层与角色相关检索，未声称复现原论文成绩。
- 只在源轨迹实际触发预定机制时纳入源失败库。E0 若无足够自然错误，流程会输出门槛失败；不会注入 Agent 错误或选择有利测试任务。
- 8/6/12 是聚类规则家族数。多个机制、随机种子和数据库副本不会被计作额外独立家族。正式功效须由开发运行估计。
