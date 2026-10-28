# d3-mini 完整路线直接动作基线

本阶段以d3-mini为主要对象，每次选择一个基础动作；预设响应和原生几何重放只证明工程链路，
不能作为模型理解图像或完成导航的证据。Qwen3.5-9B留作后续同框架对照，本阶段不部署。

## 固定合同
采用r2rce_v13_dev与r2rce-direct-v1提示词。模型看到完整指令、按时序排列的当前及最多2张旧RGB、
最近4条动作/执行/碰撞反馈；每回合重置。不使用额外memory字段、子目标或生成式规划。
深度仅供现有guard；目标、GT位置/距离/参考路径/动作不进入模型请求。
四动作前进0.25m、左右转15度、STOP；最大概率选择，并列按MOVE_FORWARD、TURN_LEFT、TURN_RIGHT、STOP顺序。
STOP要求完整路线已完成；成功仍须显式STOP且私有测地距离严格小于3m。

物理参数不变：Habitat0.3.3工程移植，RGB224/depth256，HFOV90度，相机高1.25m，
代理半径0.1m，滑动开启，深度俯仰0度，guard余量0.1m，recovery off。
最多500次决策和500个实际执行动作（含STOP）；拒绝不执行，连续3次拒绝stuck。
保留现有短历史；后续时序记忆专指新增长期或结构化记忆。旧local提示词与配置不改写。

## 模型与请求审计
目标应用vllm-sr/d3-mini，返回模型必须是d3-mini或vllm-sr/d3-mini，其他型号报model_mismatch。
submission/poll请求摘要在transport前留存；响应在JSON解码/结构校验前审计。
保留响应状态、原始字节长度和SHA256、最多64KiB脱敏正文摘要；小型JSON额外保存脱敏结构。
超长结构显式标记省略，NaN/Infinity等非有限数值以可JSON序列化的错误标记留证并拒绝执行。
HTTP错误正文继续省略，防止凭据回显。模型不符会停止当前suite，避免混合型号继续运行。
记录请求与返回型号、供应商提供的revision和用量；未提供的版本、费用记null，不能解释为0。
供应商没有公开的thinking开关；元数据think_mode为null、reasoning_mode为direct_choice，
不声称控制了模型内部推理。动作概率也不是成功概率或安全置信度。
汇总model_version_warning只比较型号字符串；false不证明供应商revision已固定。

每次实际transport之前持久预算预扣，包含POST重试、GET轮询和GET重试。
分别记录http_attempts、model_submissions、poll_attempts、retry_attempts；后者是子集，不相加。
步骤/结果/汇总保留累计计数及延迟；夹具尝试标为fixture_requests，真实http_requests仍为0。
步骤中的请求摘要保留输入文本、选项、图像URI SHA256和规范JSON payload SHA256；
帧文件单独保留，不重复写base64。该摘要不是HTTP原始字节哈希。

## 离线运行
以下从项目根目录执行，NEW替换成未使用的目录名：

```bash
HABITAT_SIM_LOG=quiet MAGNUM_LOG=quiet OFFLINE=1 .venv/bin/python -m jev_navi.offline_validation \
  --provider d3 --fixtures tests/fixtures/native_route_controls.json \
  --scenes-root data/scene_datasets --output runs/d3-mini-direct-NEW
```

两条冻结软件路线起点距目标约15m，三种控制分别为过早STOP、到达后STOP、到达不STOP，
每种含full/no_guard，共12回合，6项no_guard几何断言。参考动作和GT均为特权夹具，
不会发给模型；d3_fixture只是HTTP响应源，动作仍由生产D3Policy构造请求、解析概率选出。
full保留原guard，对照的阻断与失败必须原样展示，不降低阈值来制造成功。
不包含StartLux分层控制；原驱动默认provider=startlux及其四种控制仍保持兼容。

configs/r2rce_d3_offline.yaml提供单次STOP协议夹具示例，不是长路线策略。
configs/r2rce_d3_template.yaml是真实通道的默认关闭模板；当前运行会在请求前拒绝。
现有free_quota_confirmed标志保持false，不代表d3服务免费或已批准费用。
真实预算保持0；历史483/500关闭。后续必须单独核实供应商费用并批准阶段额度。

## 研究限制与后续顺序
正式R2R-CE标注及dense GT已准备，MP3D场景仍等待用户申请审核。
SR/SPL/NE/OS/nDTW沿用现有实现；缺失指标报告null及原因，不新增SDTW/校准/统计模块。
已暴露的两个公共软件场景不能替代正式留出评测；Habitat0.3.3不表述为官方历史环境复现。

真实数据/预算就绪后先做train诊断。未来Qwen在独立GPU部署并关闭thinking，
复用相同输入、提示词语义、执行器、guard、历史和评测。HTTP轮询成本与决策机会分别记录；
服务错误、预算截断和guard主导结果时不归因于模型能力。
只有Qwen SR高至少10个百分点、考虑场景相关性的配对区间排除0且审计支持，
才设计空间动作编码和时序记忆消融；否则保留不确定结论。

## 验收与审查记录

全量226项测试、Ruff/55文件格式检查及双环境pip check通过。
独立只读审查发现两项重要审计问题：非有限响应数值导致已开始回合保存失败、
损坏或非对象JSON响应缺少请求/响应审计。13项新增回归先复现失败，再一轮修复转绿，
包括submission/poll、异常usage、脱敏和正文上限。没有遗留代码审查项。
审查者只运行两个离线MockTransport探针；全量检查与原生验收由实施者执行。

初验目录runs/d3-mini-direct-initial-20261028：12回合、6项几何断言通过、
4个full到达类控制被原guard阻断、366次夹具提交、0真实模型HTTP、预算未变。
这份初验记录dirty=true，不作为最终源码身份验收。
最终干净提交复跑目录为runs/d3-mini-direct-final-20261028；
以其中control-summary.json记录的Git SHA、dirty=false、计数、预算与verified字段为最终证据。
