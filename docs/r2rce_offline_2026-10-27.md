# R2R-CE接口与StartLux分层离线原型 — 2026-10-27

## 本轮交付与研究边界

现有项目增加了官方R2R_VLNCE_v1-3数据导入、分版本物理协议、500步控制、
私有评测、有限语义记忆和持久调用预算。StartLux-only分层方法已实现结构化请求路径；
本轮全部使用预设响应，未检验StartLux的真实分段、进度判断或导航能力。

保留直接四动作策略及schema v1局部任务。此前真实六任务实验仍为0/6；
旧三个场景全部视为开发暴露。上轮“简单接近优先”的计划保留为历史记录，
本轮按用户最新选择改为完整R2R-CE优先，保留以后接离散R2R的接口边界。

## 数据、执行和策略边界

- prepare-r2rce读取JSON/gzip；完整保留原始指令、scene_id、起始xyzw朝向、
  trajectory_id与官方split。scenes-root对应data/scene_datasets，防止路径逃逸。
- schema v2中的evaluation集中保存目标、参考路径、dense GT和来源距离。
  同一episode的repeat_id以外定义必须一致，冲突导入拒绝。
  TaskSpec用于初始化元数据分组，不是允许传入模型的字段白名单；
  start_position、start_rotation及scene路径也不进入PolicyInput或语义记忆。
- 策略输入只包含原始指令、当前/短历史RGB、执行反馈及自有有限记忆。
  评测记录先后顺序不代替字段隔离；新增canary测试直接审查真实序列化请求。
- 缺场景时prepare仍保存缺失路径并返回2；run在开始策略前拒绝。
  test split无目标时标记evaluable=false，当前不提供官方test提交器。
- annotation_verified在导入器中只表示来源记录导入，不代表独立人工视觉复核。
  source.kind=fixture明确排除模型研究证据。

## 协议与指标

协议来源固定为VLN-CE提交729d141b2ee10628061ada74dd3a5b9f70faeba5，
参数及差异保存于environment/r2rce-protocol.json。

| 协议 | 行为 |
|---|---|
| local_v1 | 原局部默认，最多60次决策，保留旧物理设置 |
| r2rce_v13_dev | 最多500执行步含STOP、0.25m/15°、允许滑动、代理半径0.1m/高1.5m |
| r2rce_v13_dev_depth30 | 独立命名的安全深度下倾变体 |

R2R-CE配置RGB224、深度256、HFOV90°、相机高1.25m，传感器独立配置。
执行步、控制决策和HTTP尝试分别统计；语义切换/拒绝不执行动作。
环境终止后禁止再step。真实Habitat测试覆盖超过61步、独立尺寸与STOP后拒绝执行；
500步边界由轻量环境测试覆盖。

当前仍运行Habitat-Lab/Sim0.3.3。导航网格构建、物理实现与原0.1.7可能不同，
不能直接作为原基准复现成绩。深度以米制轴向距离交给安全层，未提供给语义模型。
没有接入GPS、参考路径或oracle-progress传感器。

SR要求显式STOP且最终测地距离严格小于3m；SPL使用reset后的实际测地距离。
NE/OS使用执行轨迹上的测地距离，路径长度为连续三维位置的欧氏位移之和。
nDTW删除代理连续重复位置，匹配官方dense GT，采用fastdtw==0.3.4；
缺GT或依赖时记录null和原因，绝不换成稀疏reference_path或静默改exact算法。
exact模式仅供离线参考值测试。

服务错误保留在已开始回合的SR/SPL分母，计0；缺场景等开始前错误列为不可运行。
环境构造及reset失败均记started=false，关闭环境并继续后续任务；开始决策后的运行异常不伪装为初始化错误。
报告保留有效指标覆盖率、repeat_id、场景/trajectory数量、计划/实际配对数与失败数。
缺失trajectory_id时总轨迹数为null，另报已知轨迹数与缺失ID的episode数；重复运行不增加缺失episode计数。
相同任务、物理协议、think_mode、run及phase请求上限才生成方法对照；多实际模型版本不进行合并对照。
不同配置保留config_sha256身份，缺失预算元数据时只汇总不对照。请求上限为整个suite运行的上限，
不是每回合额度；相同上限也不意味着实际调用数相同，报告仍需同时检查覆盖率和失败。
局部36任务的60%/+15个百分点门槛不用于R2R-CE。

## StartLux-only分层

原指令的标点和then等词仅产生候选边界，是否拆分由StartLux的split/merge选择决定。
子目标为有偏移的原文片段，顺序覆盖全部指令；候选超过32个时明确拒绝，不截断。
32是本项目工程上限，不是已验证的供应商上限。

每个控制决策联合请求progress(continue/complete/uncertain)及四动作。
非末片段完成时只推进片段索引；不得执行针对旧片段返回的动作。
只有末片段complete且选择STOP才执行停止，其余STOP记为语义拒绝。
安全层拒绝、碰撞和短期反馈本身不直接更新完成状态。
记忆保留原文片段、完成索引、当前索引；最近执行反馈仍有界。
本版没有地图、自由生成摘要、自动回退模型或新增低层绕障控制器。

每次StartLux响应记录问题ID、请求序号及原始响应；规划与导航的实际模型版本共同进入结果，
即使首次动作请求失败也保留成功规划响应，混用版本禁止方法对照。
规划与导航使用同一StartLux请求构建和概率解析；夹具也必须匹配实际问题ID、
候选选项及图像标记。直接策略默认think_mode=never，分层模板显式auto；
后续真实方法对照须统一think_mode，不能将推理模式差异归因于分层。

## 阶段预算与离线保护

SQLite阶段台账固定在项目runs/request-ledger.sqlite3，不随输出目录变化。
事务在HTTP前预扣，重试再次预扣；超时和进程重启不返还已预扣次数。
独立run上限与阶段总上限同时检查。数据库及回滚日志为0600，均不含密钥。

历史first-round固定483/500且live=false，剩余17次不是新增授权。
r2rce-offline-v1固定0/offline。已有phase参数不可经普通重启改写；
未来实测需先确认新的额度，创建独立phase并显式启用。
真实传输必须持久预算许可；内存CallBudget只用于软件夹具。
OFFLINE=1在每次真实预算预扣时重新检查。

## 原生软件闭环与复现

首次闭环：runs/r2rce-offline-native-01。
使用三个现有公共测试场景、原有可导航起点/目标及统一的三片段软件指令。
每回合5次控制决策、3个实际动作、2次只切换片段的事件；合计18次预设响应、
0次真实模型请求、0碰撞。nDTW因没有dense GT而为null。
这些起点都在3m成功半径内，因此显示的SR/SPL只验证计算与报告路径，
不能代表接近、路线跟随或任何模型能力。几何可行也不证明语言理解。

~~~bash
cd /home/elwen/Projects/jev-navi
.venv/bin/python -m pip install -r requirements-evaluation.lock
OFFLINE=1 .venv/bin/jev-navi prepare-r2rce \
  --episodes tests/fixtures/native_route_smoke.json \
  --scenes-root data/scene_datasets --split train --source-kind fixture \
  --output runs/r2rce-offline-native-suite-NEW.json
OFFLINE=1 .venv/bin/jev-navi run --config configs/r2rce_offline.yaml \
  --suite runs/r2rce-offline-native-suite-NEW.json --output runs/r2rce-offline-native-NEW
.venv/bin/jev-navi budget-status

OFFLINE=1 .venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/python -m pip check
.habitat/bin/python -m pip check
~~~

输出拒绝覆盖，NEW须替换为未使用的名称。每次运行保存Git提交、dirty状态、
源码/配置/任务哈希及依赖版本。研究实验应提交代码或另行保存未提交差异后再运行。

## 下一步

先获得MP3D访问授权与少量train场景；目前不自动申请授权或下载。
再冻结开发任务、人工核对视觉决策标签和独立API预算，检验StartLux分段与完成判断，
比较相同推理设置/预算下的直接策略、分层策略和记忆消融。
只有真实证据不足时才另行设计生成式上层模型。
正式泛化使用未用于调参的val_unseen场景；离散R2R同样需要相应数据访问。

## 参考

- 数据格式：https://jacobkrantz.github.io/vlnce/data
- 固定协议：https://github.com/jacobkrantz/VLN-CE/blob/729d141b2ee10628061ada74dd3a5b9f70faeba5/habitat_extensions/config/vlnce_task.yaml
- 固定指标：https://github.com/jacobkrantz/VLN-CE/blob/729d141b2ee10628061ada74dd3a5b9f70faeba5/habitat_extensions/measures.py
- StartLux接口：https://bamboonode.dev/docs/ai-model/decisions/startlux

## 交付检查

新增回归覆盖重复任务冲突、跨配置覆盖与不等请求预算比较。预算单元测试自行隔离并恢复OFFLINE，
生产环境始终逐次检查离线开关。最终验收记录见本轮交付说明与忽略目录中的运行清单。


## 本次审查与最终验收

独立只读审查覆盖完整功能分支，未发现Critical；发现的模型版本漏记、reset错误中断批次、
缺失轨迹ID误计数已通过失败回归测试修复。最终代码检查为157项测试通过，Ruff及双环境pip check通过。
Git暂存审查排除了凭据、环境、数据、运行产物和第三方仓库。

最终原生验收使用新目录runs/r2rce-offline-native-verified-20261027；
suite_manifest.json记录运行时Git SHA、dirty状态和全部哈希，delivery-verification.json记录断言结果。
运行成功条件为三个场景均完成、各5决策/3执行步/2语义切换、合计18次夹具响应、
真实HTTP请求0、阶段额度未增加。该条件仍只验证软件链路，不能作为导航性能结论。
