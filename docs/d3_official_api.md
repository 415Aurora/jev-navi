# 官方 d3-mini API 接入

官方协议来源：https://decision.vllm-sr.ai/docs 。固定请求地址 https://decision.vllm-sr.ai/v1/systemone ，Bearer认证；model为vllm-sr/d3-mini，state/images/questions.action(choice/criteria)直接同步返回answers.action.probabilities。

配置：configs/r2rce_d3_official_template.yaml。policy.kind=d3,provider=official；默认inference_sh保持兼容。官方通道拒绝base_url覆盖，以避免将官方密钥误发其他端点。DECISION_API_KEY从远端.gitignored的.env加载，文件权限600；不输出密钥。不向聊天或Git写密钥。

模型输入仍是完整路线、当前+最多2张旧RGB、最近4条反馈；不包含目标坐标、dense GT、参考路径或memory。四动作及guard/评测协议不变。型号不符拒绝执行；revision/费用缺失记null，型号匹配不能证明revision固定。

真实transport前仍须持久预算预扣；OFFLINE保护保持。官方通道不自动重试/不轮询，429或服务错误直接上报，避免违反Retry-After；后续重试策略须另行设计，不能静默重复提交。响应审计复用64KiB上限、脱敏、原始字节哈希和非有限数值拒绝；请求payload哈希计算实际官方格式。

仅离线MockTransport测试。10新增测试从缺失模块失败转为通过，覆盖真实请求形状、3图与隐私边界、401/429/503单次扣费、错误型号/NaN/缺字段/错误问题类型、工厂及密钥加载、非持久预算拒绝实际网络。全量258测试通过。

真实请求为0。模板execution.mode=offline、phase_limit=0，free_quota_confirmed=false；保存密钥不代表批准任何真实调用。需要另行批准小额接口验收预算，建立独立持久phase后才允许切换live；历史483/500保持关闭，不复用剩余额度。

建议后续接口验收仅固定少量请求验证文本、单图、多图、四动作概率与延迟；这不等于批准完整24条导航实验。免费服务的配额/速率与项目授权额度是两套独立限制，均须遵守。未测试真实服务可用性、实际revision、限流额度或导航能力。
