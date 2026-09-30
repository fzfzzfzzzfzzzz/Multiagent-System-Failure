# P2 报告内容 × 收件路由诊断

本分析属于开发集探索性诊断；协议审阅报告是确定性参考干预，不是独立人工标注。结果不修改 H1–H4。

共 192 个真实 Qwen3-8B 回合，四组各 48 个；首次成功和最终成功均为 0。

## 四组总体结果

| 组别 | n | 未提交 | 目标复发 | 避免目标错误但仍失败 | 新错误 | 触及调用上限 |
|---|---:|---:|---:|---:|---:|---:|
| model_auto | 48 | 0.250 | 0.812 | 0.188 | 0.625 | 1.000 |
| model_reviewed_route | 48 | 0.167 | 0.854 | 0.146 | 0.750 | 1.000 |
| reviewed_auto | 48 | 0.229 | 0.771 | 0.229 | 0.667 | 1.000 |
| reviewed_reviewed_route | 48 | 0.250 | 0.771 | 0.229 | 0.604 | 1.000 |

## 2×2 配对因子效应

正值表示该干预提高对应指标；对未提交、复发和新错误而言，正值是更差。区间按六个源规则家族聚类 bootstrap；p 值为家族级精确符号翻转，仅作探索性描述。

| 范围 | 指标 | 因子 | 效应 | 95% CI | p |
|---|---|---|---:|---:|---:|
| all | first_success | content | +0.000 | [+0.000, +0.000] | 1.000 |
| all | first_success | routing | +0.000 | [+0.000, +0.000] | 1.000 |
| all | first_success | interaction | +0.000 | [+0.000, +0.000] | 1.000 |
| all | final_success | content | +0.000 | [+0.000, +0.000] | 1.000 |
| all | final_success | routing | +0.000 | [+0.000, +0.000] | 1.000 |
| all | final_success | interaction | +0.000 | [+0.000, +0.000] | 1.000 |
| all | abstained | content | +0.044 | [-0.073, +0.144] | 0.531 |
| all | abstained | routing | -0.045 | [-0.142, +0.045] | 0.500 |
| all | abstained | interaction | +0.104 | [-0.028, +0.222] | 0.312 |
| all | recurrence | content | -0.075 | [-0.200, +0.083] | 0.625 |
| all | recurrence | routing | +0.035 | [-0.049, +0.125] | 0.750 |
| all | recurrence | interaction | -0.028 | [-0.111, +0.056] | 1.000 |
| same | first_success | content | +0.000 | [+0.000, +0.000] | 1.000 |
| same | first_success | routing | +0.000 | [+0.000, +0.000] | 1.000 |
| same | first_success | interaction | +0.000 | [+0.000, +0.000] | 1.000 |
| same | final_success | content | +0.000 | [+0.000, +0.000] | 1.000 |
| same | final_success | routing | +0.000 | [+0.000, +0.000] | 1.000 |
| same | final_success | interaction | +0.000 | [+0.000, +0.000] | 1.000 |
| same | abstained | content | +0.010 | [-0.206, +0.211] | 1.000 |
| same | abstained | routing | -0.107 | [-0.300, +0.074] | 0.438 |
| same | abstained | interaction | +0.175 | [-0.103, +0.431] | 0.312 |
| same | recurrence | content | -0.031 | [-0.250, +0.181] | 0.812 |
| same | recurrence | routing | +0.086 | [-0.083, +0.267] | 0.500 |
| same | recurrence | interaction | -0.022 | [-0.222, +0.178] | 1.000 |
| transfer | first_success | content | +0.000 | [+0.000, +0.000] | 1.000 |
| transfer | first_success | routing | +0.000 | [+0.000, +0.000] | 1.000 |
| transfer | first_success | interaction | +0.000 | [+0.000, +0.000] | 1.000 |
| transfer | final_success | content | +0.000 | [+0.000, +0.000] | 1.000 |
| transfer | final_success | routing | +0.000 | [+0.000, +0.000] | 1.000 |
| transfer | final_success | interaction | +0.000 | [+0.000, +0.000] | 1.000 |
| transfer | abstained | content | +0.078 | [+0.000, +0.178] | 0.500 |
| transfer | abstained | routing | +0.017 | [+0.000, +0.050] | 1.000 |
| transfer | abstained | interaction | +0.033 | [+0.000, +0.100] | 1.000 |
| transfer | recurrence | content | -0.119 | [-0.272, +0.069] | 0.250 |
| transfer | recurrence | routing | -0.017 | [-0.050, +0.000] | 1.000 |
| transfer | recurrence | interaction | -0.033 | [-0.100, +0.000] | 1.000 |
| F1 | first_success | content | +0.000 | [+0.000, +0.000] | 1.000 |
| F1 | first_success | routing | +0.000 | [+0.000, +0.000] | 1.000 |
| F1 | first_success | interaction | +0.000 | [+0.000, +0.000] | 1.000 |
| F1 | final_success | content | +0.000 | [+0.000, +0.000] | 1.000 |
| F1 | final_success | routing | +0.000 | [+0.000, +0.000] | 1.000 |
| F1 | final_success | interaction | +0.000 | [+0.000, +0.000] | 1.000 |
| F1 | abstained | content | -0.125 | [-0.425, +0.200] | 0.625 |
| F1 | abstained | routing | +0.125 | [+0.025, +0.225] | 0.250 |
| F1 | abstained | interaction | +0.250 | [+0.050, +0.450] | 0.250 |
| F1 | recurrence | content | +0.075 | [-0.225, +0.375] | 1.000 |
| F1 | recurrence | routing | -0.125 | [-0.325, +0.000] | 0.500 |
| F1 | recurrence | interaction | -0.050 | [-0.150, +0.000] | 1.000 |
| F2 | first_success | content | +0.000 | [+0.000, +0.000] | 1.000 |
| F2 | first_success | routing | +0.000 | [+0.000, +0.000] | 1.000 |
| F2 | first_success | interaction | +0.000 | [+0.000, +0.000] | 1.000 |
| F2 | final_success | content | +0.000 | [+0.000, +0.000] | 1.000 |
| F2 | final_success | routing | +0.000 | [+0.000, +0.000] | 1.000 |
| F2 | final_success | interaction | +0.000 | [+0.000, +0.000] | 1.000 |
| F2 | abstained | content | +0.250 | [+0.062, +0.438] | 0.250 |
| F2 | abstained | routing | -0.125 | [-0.375, +0.062] | 0.750 |
| F2 | abstained | interaction | +0.125 | [+0.000, +0.250] | 0.500 |
| F2 | recurrence | content | -0.250 | [-0.438, -0.062] | 0.250 |
| F2 | recurrence | routing | +0.125 | [-0.062, +0.375] | 0.750 |
| F2 | recurrence | interaction | -0.125 | [-0.250, +0.000] | 0.500 |
| F3 | first_success | content | +0.000 | [+0.000, +0.000] | 1.000 |
| F3 | first_success | routing | +0.000 | [+0.000, +0.000] | 1.000 |
| F3 | first_success | interaction | +0.000 | [+0.000, +0.000] | 1.000 |
| F3 | final_success | content | +0.000 | [+0.000, +0.000] | 1.000 |
| F3 | final_success | routing | +0.000 | [+0.000, +0.000] | 1.000 |
| F3 | final_success | interaction | +0.000 | [+0.000, +0.000] | 1.000 |
| F3 | abstained | content | +0.125 | [-0.042, +0.250] | 0.375 |
| F3 | abstained | routing | -0.125 | [-0.250, +0.042] | 0.375 |
| F3 | abstained | interaction | +0.083 | [-0.333, +0.417] | 1.000 |
| F3 | recurrence | content | -0.167 | [-0.333, +0.042] | 0.312 |
| F3 | recurrence | routing | +0.083 | [-0.083, +0.208] | 0.625 |
| F3 | recurrence | interaction | +0.000 | [-0.333, +0.333] | 1.000 |
| F4 | first_success | content | +0.000 | [+0.000, +0.000] | 1.000 |
| F4 | first_success | routing | +0.000 | [+0.000, +0.000] | 1.000 |
| F4 | first_success | interaction | +0.000 | [+0.000, +0.000] | 1.000 |
| F4 | final_success | content | +0.000 | [+0.000, +0.000] | 1.000 |
| F4 | final_success | routing | +0.000 | [+0.000, +0.000] | 1.000 |
| F4 | final_success | interaction | +0.000 | [+0.000, +0.000] | 1.000 |
| F4 | abstained | content | -0.042 | [-0.375, +0.250] | 1.000 |
| F4 | abstained | routing | -0.042 | [-0.167, +0.083] | 1.000 |
| F4 | abstained | interaction | -0.083 | [-0.333, +0.167] | 1.000 |
| F4 | recurrence | content | +0.042 | [-0.250, +0.375] | 1.000 |
| F4 | recurrence | routing | +0.042 | [-0.083, +0.167] | 1.000 |
| F4 | recurrence | interaction | +0.083 | [-0.167, +0.333] | 1.000 |

## 互斥失败构成

| 类别 | 回合数 | 比例 |
|---|---:|---:|
| target_recurrence_failure | 147 | 0.766 |
| abstained | 43 | 0.224 |
| other_failure_without_target_recurrence | 2 | 0.010 |

## 解释

- 更好的参考内容和预设收件角色都没有产生一次完整成功，因此 P2 不支持把零成功主要归因于报告措辞或初始收件人选择。
- 参考内容对目标复发只有小幅总体下降；需要结合未提交和其他错误判断，不能把少复发直接解释成学习成功。
- 3 次调用的形式约束可解，但真实模型普遍触及预算上限；剩余瓶颈是跨角色执行与时序控制，而不是单独的报告内容或路由。
- 下一步若继续，只应在开发集上分离执行预算/能力与记忆干预，例如固定最佳 P2 单元后比较 3/5/7 次调用；不得用本结果回改冻结 H1/H4。
