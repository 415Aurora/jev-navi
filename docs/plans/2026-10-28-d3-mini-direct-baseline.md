# d3-mini 多模态直接导航实施计划

日期：2026-10-28；用户已确认执行。基线 e21216e1608a2a2a92afe1c349ed6f9fcb9f7d33。
分支 codex/d3-mini-direct-baseline；复用远端现有目录和环境。

## 约束与验收合同
以d3-mini为主要研究对象；Qwen3.5-9B的适配/部署留到后续，不下载权重，不申请GPU作业。
不增加空间动作编码、长期记忆、语义分层或混合规划；不训练、不下载MP3D，不合并main。
历史预算483/500关闭；本阶段0/offline，真实模型HTTP必须为0。
完整原始指令、当前+最多2张历史RGB、最近4条执行反馈；每回合重置。
深度仅用于安全层，目标/真值位姿/距离/参考路径/dense GT/oracle动作不进入模型请求。
固定四动作0.25m/15度/STOP，概率argmax并列沿用枚举顺序。
新增r2rce-direct-v1完整路线提示词，保留旧local提示词。
沿用r2rce_v13_dev、500次决策与500执行步、三次连续拦截stuck、recovery off。
SR=显式STOP且测地距离严格小于3m；服务失败在已开始回合分母；SR/SPL/NE/OS/nDTW保持口径。

## 实施步骤
- [ ] Task 1：共享直接动作上下文；D3实际模型身份校验、原始响应留存、提交/轮询/重试计数。
- [ ] Task 2：d3_fixture使用MockTransport与生产适配器；接入runner/报告；固定路线协议与默认关闭模板。
- [ ] Task 3：现有原生路线三控制×两条件×两场景=12回合；6项无拦截几何断言；保留guard差异。
- [ ] Final：全量测试、Ruff/格式、双环境pip check；一次独立只读全分支审查，重要问题一轮TDD修复。
- [ ] Delivery：按独立改动提交；最终干净SHA原生复跑，核对0真实HTTP和预算不变；推送并核对SHA/main/标签。

先写测试，观察缺失行为失败，再做最小实现。现有并发预算、终止边界、失败审计测试复用并全量回归。

## 接口与数据流
PolicyInput和Decision不变；共享action_context生成state/images/question/criteria。
D3Policy增加prompt_version选择；旧默认仍为local-navigation-v3-recovery。
新增policy.kind=d3_fixture读取显式HTTP交换(method/path/status_code/json)记录；不直接返回动作。
run_route_controls与CLI增加provider=startlux|d3，默认startlux不变；d3只运行三种直接控制。
请求摘要记录文本、问题、图像URI哈希及payload哈希，不重复保存base64；响应保留供应商原始结构。
请求审计为http_attempts/model_submissions/poll_attempts/retry_attempts，重试是尝试子集，不能相加。
夹具的尝试记为fixture_requests，真实http_requests=0；缺版本/费用记null。

## 审查重点
旧提示词兼容、错误模型响应仍可审计、异步/重试在实际transport前扣费、
MockTransport不能进入真实网络或研究对照、终止/500步后不执行、每回合历史和夹具游标清空、
失败/部分结果仍保存、原生引用路线hash/代码SHA/dirty身份完整。

## 后续条件
MP3D授权与原生场景验证、新调用预算明确后才做train小规模真实诊断。
未来Qwen关闭thinking并共享导航系统；不同供应商轮询开销单列，预算截断不能充当完整公平对照。
Qwen SR至少高10个百分点、场景相关性配对区间排除0且非系统/服务/预算因素主导，
才进入空间动作编码与时序记忆的独立消融设计。本阶段不实现该统计分析或启动模型比较。
