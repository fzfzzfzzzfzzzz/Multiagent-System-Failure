# H1/H4 独立测试确认结果

本文件只解释预先冻结的 same 条件主比较。首次成功差值为正、复发率差值为负才是预期方向。六项主检验共同进行 Holm 校正。

| 假设 | 对照（A−B） | 指标 | 家族数 | 差值 | 95% CI | 原始 p | Holm p | 证据 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| H1 | components_SOCDKPU − components_SP | first_success | 12 | 0.0000 | [0.0000, 0.0000] | 1 | 1 | no_observed_difference |
| H1 | components_SOCDKPU − components_SP | recurrence | 12 | -0.0949 | [-0.2361, 0.0440] | 0.2676 | 1 | inconclusive |
| H4 | matched − cycle | first_success | 12 | 0.0000 | [0.0000, 0.0000] | 1 | 1 | no_observed_difference |
| H4 | matched − cycle | recurrence | 12 | -0.0139 | [-0.1134, 0.0880] | 0.8184 | 1 | inconclusive |
| H4_validation | matched − random | first_success | 12 | 0.0000 | [0.0000, 0.0000] | 1 | 1 | no_observed_difference |
| H4_validation | matched − random | recurrence | 12 | -0.0324 | [-0.1227, 0.0509] | 0.5391 | 1 | inconclusive |

## 判定

- **H1**：`not_supported_by_this_confirmation`
- **H4**：`not_supported_by_this_confirmation`
- **H4_validation**：`not_supported_by_this_confirmation`

## same 条件绝对结果

| 实验 | 方法 | 回合数 | 首次成功率 | 复发率 | 平均总 tokens |
|---|---|---:|---:|---:|---:|
| E1 | components_SOCDKPU | 135 | 0.0000 | 0.7185 | 41861.3 |
| E1 | components_SP | 135 | 0.0000 | 0.8222 | 22064.5 |
| E3 | cycle | 135 | 0.0000 | 0.6889 | 25183.1 |
| E3 | matched | 135 | 0.0000 | 0.6667 | 25390.1 |
| E3 | random | 135 | 0.0000 | 0.7037 | 24779.0 |

## 预先保留的 transfer 条件分析

这些结果不改变 same 条件主判定。这里的 Holm p 来自各实验输出的完整注册比较族。

| 实验 | 对照（A−B） | 复发率差值 | 95% CI | Holm p |
|---|---|---:|---:|---:|
| E1 | components_SOCDKPU − components_SP | -0.2130 | [-0.2894, -0.1366] | 0.009766 |
| E3 | matched − cycle | -0.1875 | [-0.2778, -0.0972] | 0.1094 |
| E3 | matched − random | -0.1227 | [-0.2060, -0.0462] | 0.375 |

该判定限于 Qwen3-8B、当前受控 workflow 环境、冻结任务与预算。没有预注册非劣效界限，区间跨零时不能声称等效或非劣。transfer/reversal 与成本结果保留在导出的完整分析中，作为条件分析而不改变主判定。
