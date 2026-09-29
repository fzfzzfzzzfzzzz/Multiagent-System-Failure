# 基线实现与文献边界

G-Memory 原论文：[G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems](https://arxiv.org/html/2506.07398v2)，实现时核对了其分层经验及角色定制设计。

本项目 `gmemory_adapt` 是明确标注的 **G-Memory-inspired adaptation**：先检索任务记录，沿相近任务扩展，再提取经验层信息与角色相关的交互证据。使用词项重叠检索与显式证据引用，没有复现原实现的全部图更新、提示词、嵌入模型或基准。对照表必须保留 adaptation 字样，不能写“超过原版 G-Memory”。

`independent` 为相同底模按角色自身可见轨迹各自生成反思；它是独立反思基线，不自称原版 Reflexion。

`mechanism` 使用共用报告器预测的相关角色，不读取 gold 机制标签。`culprit` 使用同一报告器的预测责任角色，允许为空。`learned` 的候选包括空集、四个单角色、角色对、三角色组合及广播，包含角色对特征；训练来自实际配对行为结果。

STAR、H-RePlan、MANTA、EvoMAC、COOP² 不以简化提示词冒充复现。它们仍属于原研究方案的相关工作和可选恢复/组织调整扩展，未加入本次必须运行的主对照。

SQLite 工作流独立执行数据库读写与事务日志，但与受控工作流共享业务规则。可报告“可执行应用工作流验证”，不能用这一实现替代 CooperBench-derived 或生产环境的外部有效性证据。
