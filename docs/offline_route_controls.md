# 半径外起步的离线路线对照

这是特权软件回归：参考动作由原生场景的greedy follower离线生成，预设响应经过真实StartLux请求构造与概率解析。
结果不测量模型的语言理解或导航能力。两个目标来自现有城堡/公寓调查点，调查测地距离约15.44m/15.71m；
每次运行仍重新检查实际reset距离必须严格大于3m。

## 生成冻结夹具

在远端项目根目录，用现有Python3.9原生环境生成：

~~~bash
HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet .habitat/bin/python scripts/author_route_controls.py \
  --catalog configs/offline_route_catalog.json --scenes-root data/scene_datasets \
  --output runs/native-route-controls-NEW.json
~~~

目标半径保持3m；参考跟随器使用2.5m停止余量。它在生成后reset并重放动作，保存每步实际位置作为
软件dense GT，并记录场景/脚本/worker哈希及环境版本。此GT不是官方R2R-CE标注。
输出拒绝覆盖；NEW替换成未使用名称。文件可在人工审核后作为小型软件夹具纳入Git；
正式实验数据和运行结果继续留在忽略目录。

## 运行四种对照

~~~bash
OFFLINE=1 .venv/bin/python -m jev_navi.offline_validation \
  --fixtures runs/native-route-controls-NEW.json --scenes-root data/scene_datasets \
  --output runs/offline-route-controls-NEW
~~~

每条路线运行四种控制，各含full/no_guard两种条件，共八回合：

| 控制 | SR | OS | 结束规则 |
|---|---:|---:|---|
| 原地立即STOP | 0 | 0 | 起点必须在3m外 |
| 完整参考动作后STOP | 1 | 1 | NE严格小于3m |
| 到达后不STOP | 0 | 1 | 决策预算耗尽 |
| 分层完成第一片段再STOP | 1 | 1 | 一次只推进语义、不执行旧动作 |

上述数值是no_guard几何对照的预期。full保留已有深度检查：如果发生拒绝，仍保存实际轨迹、
SR/OS/nDTW和阻断数，单独标记guard_deviation，不能将该回合描述为导航成功。
四种无拦截几何断言全部通过且阶段预算未变化时，该路线的软件对照才通过。
到达类软件轨迹的nDTW应接近1，SPL需位于(0,1]；这些值只证明参考重放/计算链路一致。
纯合成环境测试使用exact DTW作为黄金参考；原生运行使用已固定的FastDTW模式。

## 产物与边界

control-summary.json记录断言结果、各条件真实指标、Git身份、依赖、实现与清单哈希、实际HTTP计数、
夹具响应次数和预算是否变化；control-report.md展示对照表。子运行保留原始清单、图像、深度、响应事件、
配置/任务/协议哈希与回放。任何模型HTTP或策略错误都会令对照失败。
当前阶段保持r2rce-offline-v1的0额度；所有策略固定为MockTransport软件夹具。
文件级正式数据检查见[r2rce_data_readiness.md](r2rce_data_readiness.md)。

## 2026-10-28 原生初验

远端全量190项测试、Ruff及双环境依赖检查通过。初验目录为runs/offline-route-controls-initial-20261028，
包含两个场景各四种控制、full/no_guard两种条件，共16回合。8项no_guard几何对照全部符合预期。
真实模型HTTP为0；554次为预设响应，阶段台账保持历史483/500关闭、本阶段0/offline。
这些是软件断言结果，不是模型研究成绩；几何对照本来就包含预期失败的过早STOP和不STOP回合。

| 场景 | 初始测地距离 | 参考STOP后距离 | 参考动作数 | 原生参考碰撞 |
|---|---:|---:|---:|---:|
| 城堡 | 15.444864m | 2.264287m | 58 | 0 |
| 公寓 | 15.714267m | 2.392133m | 82 | 0 |

两条参考路线在关闭深度拦截时均能重放。到达+STOP及分层STOP回合SR=1；到达但不STOP回合SR=0、OS=1；
这些到达类参考重放nDTW=1。开启guard的6个到达类回合均被拦截并以stuck结束，实际SR=0。
结果全部保留，不通过改阈值、半径或预期来隐藏这一差异。独立只读审查发现的失败回合漏记、
预算异常报告缺失已由7项失败回归修复。

首次阻断前两种条件的动作和位置完全一致；无拦截参考的下一步均约0.25m且没有原生碰撞。
阻断深度像素到理想直行中心线段的最小平面距离分别约0.240m/0.210m，
代理半径0.1m、guard额外余量0.1m。这显示当前矩形走廊启发式存在保守性；
它不是精确扫掠体，也不能据单次原生碰撞结果证明现实安全。此轮保持安全层行为不变。
详细离线审计保存于runs/r2rce-offline-validation-development/guard-disagreement-audit.json。

冻结软件参考保存于tests/fixtures/native_route_controls.json，记录生成时的干净源码提交、
脚本/worker/场景哈希及实际GT；授权数据和实验产物继续放在忽略目录。
最终从干净提交复核的目录为runs/offline-route-controls-final-20261028；
以其control-summary.json的Git身份、verified、budget和各回合字段作为最终证据。
