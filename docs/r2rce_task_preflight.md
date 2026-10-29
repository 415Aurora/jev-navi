# R2R-CE 开发任务冻结与预检

2026-10-29。首批只冻结 train 开发任务；不挑选 val_unseen 任务，不使用模型结果或当前场景可用性决定抽样。

## 已冻结的清单

Git 文件 `configs/r2rce_train_dev24.json` 固定24条任务：4个场景，每场景6条不同 trajectory，每条 trajectory 只取一条原始指令。场景如下：

- `759xd9YjKW5`
- `GdvgFV5R1Z5`
- `SN83YJsR3w2`
- `aayBHfsNo7d`

算法为 `train-scene-trajectory-sha256-v1`，seed 为 `jev-train-dev-v1`。分别以 `[version, seed, kind, identities...]` 的规范 JSON SHA256 排序场景、轨迹和episode；先筛出至少6条不同轨迹的场景，再取哈希最小的4个，每个场景取6条轨迹、每轨迹取1条指令。输入排序不影响结果，不根据是否下载或起点距离筛选。

清单记录原始标注/GT字节哈希、协议、ID/scene/trajectory/repeat身份和任务定义哈希。完整任务原文、初始位姿、目标、dense GT只在忽略的派生suite中，不复制进Git清单。定义哈希包含初始化、指令和私有评测字段，排除可迁移的绝对场景文件路径。当前源文件路径记录于suite；迁移时重新prepare，原始文件内容必须保持一致。

校验会重新导入当前原始文件并重算选择，检查来源、任务定义、成员及顺序。它不是数字签名：如果人为同时修改算法参数与清单，那就是新的实验定义，应通过Git审查，不能冒称原实验。

## 实际文件检查结果

远端派生目录：`data/r2rce-source/frozen-train-dev24-v1/`，含suite、train与val_unseen报告、冻结收据。

| 范围 | episodes | 场景 | 不同轨迹 | dense GT | 缺失场景 |
|---|---:|---:|---:|---:|---:|
| train开发清单 |24|4|24|24/24|4|
| 完整val_unseen只读预检 |1839|11|613|1839/1839|11|

train与val_unseen场景集合无交集。以上两者均 `ready=false`，原因是MP3D场景尚未提供；不是模型导航失败。原始文件哈希通过，train冻结定义通过。所有检查始终 `geometry_verified=false`、`live_authorized=false`、真实HTTP为0。未读取留出集结果做调参或选任务。

文件可读并不等于场景能加载/导航；本次没有运行MP3D原生几何检查，也没有运行新的导航闭环。已有原生夹具闭环属于前阶段工程验证，其性能不能外推到这些任务。

## 复现

项目根目录已有原始标注、GT与完整prepared train清单时，用全新输出路径：

```bash
OFFLINE=1 .venv/bin/jev-navi freeze-r2rce \
  --suite data/r2rce-source/prepared-20261028/train-suite.json \
  --output runs/train-dev24-NEW/suite.json \
  --manifest runs/train-dev24-NEW/manifest.json
OFFLINE=1 .venv/bin/jev-navi check-r2rce \
  --suite runs/train-dev24-NEW/suite.json \
  --selection configs/r2rce_train_dev24.json --require-ndtw
```

freeze成功返回0，只表示定义冻结；check在当前缺场景时返回2，报告仍输出。输出文件已存在时拒绝覆盖。可用已有 `prepare-r2rce` 重新生成完整来源清单，具体源路径见数据接入文档。

嵌入selection的suite在check和run入口自动验证，run验证发生在预算、policy、环境及输出目录创建前；外部Git清单可通过check的 `--selection` 再次校验。旧schema1和未冻结schema2不强制使用新清单，保持历史工具兼容；正式使用冻结实验前必须保留selection并对Git清单运行check。

固定参数和源哈希使重建可审计；更换seed、数量、指令、GT、起始位姿或成员不属于同一冻结实验。真实运行仍需场景到位、原生几何验收、配置/模型版本冻结，以及新的明确预算授权。历史483/500保持关闭，本阶段0/offline；不下载MP3D、不部署模型、不训练、不合并main。
