# 首次官方d3-mini train导航诊断

用户明确批准最多200真实HTTP尝试，失败计数，不自动重试；两不同train场景各一条冻结任务，每条最多100次决策。只运行full，保留guard，物理协议r2rce_v13_dev和500执行步上限不变。100次上限触发为诊断截断，不充当完整500步基准成绩。

任务按冻结24条清单顺序，取前两个不同场景各自第一条，不根据结果、指令或距离挑选。先对完整父suite执行原始来源重导入及Git清单验证，再复制两条任务。派生suite不冒称完整24条冻结selection，单独保留diagnostic_parent_selection父清单哈希/选择规则和ID；由run manifest记录完整子suite哈希。

配置configs/r2rce_d3_train_diagnostic.yaml；输入及guard/STOP判定保持。独立phase d3-train-diagnostic-20261103上限200，历史phase不变，任务结束后关闭，不补跑。真实服务版本未固定，费用未提供记未知。仅train诊断，不涉及val_unseen，不作公平对照或泛化结论。

产物runs/d3-train-diagnostic-20261103/：suite.json、原始run日志、steps与帧、manifest和报告。用户未跟踪的.vscode目录未改动或提交；若Git身份dirty=true，仅IDE目录未跟踪状态也会导致此标记，不将它伪装为干净。导航源码仍与已验收1a6c5bf一致。
