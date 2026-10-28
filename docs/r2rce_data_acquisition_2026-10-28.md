# R2R-CE v1-3 标注与 GT 接入记录

日期：2026-10-28（Asia/Shanghai）。MP3D访问申请已由用户发送，仍等待审核。
本次只获取[官方数据页](https://jacobkrantz.github.io/vlnce/data)公开链接提供的标注包与预处理包，
没有下载MP3D场景，没有运行仿真、训练或模型调用。

## 来源与完整性

原始包及选择性解压文件保存在远端 `/home/elwen/Projects/jev-navi/data/r2rce-source/`。
两个包均通过ZIP CRC检查；归档与选用文件的SHA256、下载链接、数量及派生清单身份
记录在[数据来源清单](../environment/r2rce-data-manifest.json)。SHA256由本地计算，
用于后续复现和变更检测；官方页面没有提供发布方校验和。

本次下载使用已运行的Xray代理 `socks5h://127.0.0.1:10808`，只对curl命令设置代理，
没有修改系统或v2rayN配置。下载过程中使用 `.part` 后缀，ZIP验证后才改为正式文件名。
普通包约2.6MB，预处理包约261.6MB。预处理包只解压四个原始split及三个GT文件；
envdrop、joint_train_envdrop与embeddings未解压、未使用。

## 元数据与导入验收

| split | episodes | scenes | 有dense GT的episodes | 当前缺失场景 |
|---|---:|---:|---:|---:|
| train | 10819 | 61 | 10819 | 61 |
| val_seen | 778 | 53 | 778 | 53 |
| val_unseen | 1839 | 11 | 1839 | 11 |
| test | 3408 | 18 | 0 | 18 |

所有split合计涉及90个不同场景。普通包与预处理包的episode ID、场景、起始位姿、目标、
参考路径、trajectory、info及指令原文逐条一致；token/vocabulary差异不属于任务定义。
仅做机械一致性检查，没有用val_unseen指令或GT调参。

用现有 `prepare_r2rce` 和 `check_r2rce` 函数分别处理四个split，派生文件位于
`data/r2rce-source/prepared-20261028/{split}-suite.json` 与 `{split}-readiness.json`。
检查确认源文件哈希匹配、schema合法、非test任务全部具备目标及dense GT、
GT动作不保留在导入清单中，私有评测信息保存在evaluation字段。
所有split的 `ready=false`：场景尚未提供；test还缺少公开目标与GT，不能进行本地SR/SPL/nDTW评测。
文件清单不会把这些任务记作模型导航失败。

预检报告始终 `geometry_verified=false`、`live_authorized=false`。
模型HTTP为0，持久请求台账文件及其伴随文件的SHA256在导入前后完全一致。
数据导入/预检相关32项测试通过，全量190项测试及Ruff/50文件格式检查通过。
另用实际CLI复查val_seen：778条GT覆盖完整、53个场景缺失，按约定返回2，模型HTTP为0。
该验收覆盖文件和结构，不替代原生场景可达性验收或官方评测环境复现。

## 复现入口

在远端项目根目录，已有原始文件时，可使用新的输出文件名重建train清单：

```bash
OFFLINE=1 .venv/bin/jev-navi prepare-r2rce \
  --episodes data/r2rce-source/R2R_VLNCE_v1-3/train/train.json.gz \
  --gt data/r2rce-source/R2R_VLNCE_v1-3_preprocessed/train/train_gt.json.gz \
  --scenes-root data/scene_datasets --split train --source-kind official \
  --output runs/r2rce-train-prepared-NEW.json
OFFLINE=1 .venv/bin/jev-navi check-r2rce \
  --suite runs/r2rce-train-prepared-NEW.json --require-ndtw
```

缺场景时prepare仍保存清单并返回2；随后预检也返回2，这是当前预期。
val_seen/val_unseen替换split和对应文件；test不传GT且当前无法用于本地指标评测。
原始来源文件需保留，否则源哈希预检失败；迁移路径后重新prepare。
本次批量操作直接调用相同Python函数，使用紧凑JSON保存大清单；
CLI保存格式可能不同，因此派生文件字节哈希只对应实际记录的那次生成，不能据此要求不同序列化结果字节相同。

## MP3D审核通过后的顺序

先核对用户获批的访问范围和所需文件，再按空间选择train开发场景；场景应保留
`data/scene_datasets/mp3d/<scan>/<scan>.glb`布局。当前约25GB可用空间，不能默认容纳全部场景。
场景放置后重新运行文件预检、原生reset与可达性检查，并冻结开发/留出任务。
真实模型实验仍须获得新的明确调用预算；历史483/500关闭，本阶段0/offline。
数据和运行产物不进入Git，main与基线标签保持不变。
