# JEV-Navi：冻结多模态决策模型的局部导航实验

本项目实现语言目标 → RGB 与短历史 → 冻结模型动作提议 → 米制深度拦截 → Habitat 单步执行的闭环。首阶段研究局部语义导航；完整 VLN、R2R-CE 和 G1/SONIC 迁移留待后续验证。

**脚本、回放、合成环境和预设API响应只用于软件验证。此前真实StartLux六任务实验均为0/6；本轮离线原型不提供新的模型导航成绩。**

## 当前阶段：完整路线接口与离线分层原型

现已接入R2R-CE v1-3格式，增加500步控制、独立RGB/深度尺寸、私有评测和StartLux-only有限语义分层。
当前运行环境仍是Habitat0.3.3工程移植；官方0.1.7复现和真实MP3D评测尚未完成。
原三个测试场景属于已暴露开发夹具，不是新的留出场景。

~~~bash
# 上游nDTW采用固定fastdtw实现；无需下载模型
.venv/bin/python -m pip install -r requirements-evaluation.lock

# 原生三场景软件闭环：使用预设响应，真实模型请求为0
OFFLINE=1 .venv/bin/jev-navi prepare-r2rce \
  --episodes tests/fixtures/native_route_smoke.json \
  --scenes-root data/scene_datasets --split train --source-kind fixture \
  --output runs/r2rce-offline-native-suite-NEW.json
OFFLINE=1 .venv/bin/jev-navi run --config configs/r2rce_offline.yaml \
  --suite runs/r2rce-offline-native-suite-NEW.json --output runs/r2rce-offline-native-NEW
.venv/bin/jev-navi budget-status
~~~

正式数据导入时换成官方episodes与可选的dense GT文件，保持scenes-root为data/scene_datasets。
缺场景会输出精确路径；prepare命令仍保存清单并返回2，run在策略运行前拒绝缺失资源。
没有GT时nDTW为null并报告覆盖率，不以稀疏reference_path代替。
可用 `check-r2rce --suite PATH [--require-ndtw]` 复查源哈希、场景文件、目标及GT覆盖率；
该命令不初始化预算、不发送模型请求。见[正式数据预检](docs/r2rce_data_readiness.md)。

原首轮预算固定记录483/500且关闭；新阶段0/offline。真实请求必须使用显式live配置和持久阶段额度，
free_quota_confirmed单独设为true不会打开网络。旧探针使用内存预算时会被请求层拒绝。
完整设计、限制及复现说明见[离线阶段说明](docs/r2rce_offline_2026-10-27.md)。

## 半径外起步的离线正反对照

新增两个公开测试场景的长路线软件夹具，起点距目标约15m。四种控制分别检查过早STOP、
到达后STOP、到达不STOP及分层推进；full保留安全层，no_guard仅作特权几何控制。
初始化和失败回合均保存审计，报告不得将预设响应解释为真实模型能力。
生成与运行步骤见[离线路线对照](docs/offline_route_controls.md)。

## d3-mini 直接动作基线

研究对象收敛为d3-mini，新增完整路线提示词、生产适配器的HTTP交换夹具与独立请求计数。
当前只执行零真实模型请求的软件验收；Qwen部署和正式实验留到后续。
固定协议、运行命令和研究限制见[d3-mini方案](docs/d3_mini_direct_baseline.md)。

## 环境与快速运行

决策侧使用 Python ≥3.11（本机验证为 3.12.3）。官方 Habitat 的兼容环境独立放在 `.habitat`，通过 JSON 子进程接口通信；无需激活或修改其他项目环境。

```bash
cd /home/elwen/Projects/jev-navi
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
bash scripts/setup_habitat.sh
.venv/bin/jev-navi doctor

# 不依赖 Habitat 的控制循环软件检查
.venv/bin/jev-navi smoke --backend synthetic --output runs/synthetic-smoke

# 真实 Habitat-Lab + Habitat-Sim：两条软件轨迹 × 三种输入条件，共六回合
.venv/bin/jev-navi run --config configs/smoke.yaml \
  --suite configs/smoke_tasks.yaml --output runs/habitat-smoke
.venv/bin/jev-navi report runs/habitat-smoke

# 单条真实场景检查，可替换明确的起点与目标
.venv/bin/jev-navi smoke --backend habitat \
  --scene data/scene_datasets/skokloster-castle.glb \
  --start 3.4107175 0.1335671 8.3882465 \
  --goal 3.4107175 0.1335671 7.8882465 --output runs/single-habitat-smoke
```

决策侧的实际依赖版本保存在 `requirements-policy.lock`（本机 Python 3.12）；需要重建相同版本时先安装该锁文件，再执行 `pip install --no-deps -e .`。

每次使用新的输出目录，程序拒绝覆盖已有结果。可通过环境变量 `JEV_HABITAT_PYTHON` 或配置 `environment.python` 指定其他 Habitat 解释器。

实际安装为 **Habitat-Sim 0.3.3 + Habitat-Lab 0.3.3 + Python 3.9.23**。计划中的 0.3.4/Python 3.11 组合存在官方包与依赖兼容问题，详情、锁文件和源提交见 [环境说明](docs/environment.md) 与 [版本清单](environment/source-manifest.json)。不把本环境的结果直接等同于原始 Habitat 0.1.7 的 R2R-CE 结果。

## 真实模型接口（历史基线与后续启用条件）

已实现 StartLux/Bamboo 与 inference.sh d3-mini 两种适配器。输入图像为 PNG data URI，严格按旧到新排列。模型选择取四种动作概率的最大值，保存完整原始响应；不把概率解释为导航安全置信度。

已核对 [Bamboo 官方 StartLux 文档](https://bamboonode.dev/docs/ai-model/decisions/startlux)（2026-10-27）：模型为 `startlux-decision-27b-autothink`，同步接口为 `https://api.bamboonode.dev/v1/decisions`，`/v1/systemone` 是兼容别名。当前测试期免费，零余额新账号无需充值；API Key 必须拥有 `default` 分组和该模型的访问权限。免费政策可能变化，运行前核对控制台。直接策略默认使用 `think_mode=never`，分层模板使用 `auto`，均设置 `allow_none=false`，每张图像在 state 中有按时间排列的 `<image>` 标记。

在 [控制台](https://bamboonode.dev/console) 创建 API Key。本地测试推荐持久化保存：

```bash
.venv/bin/jev-navi configure-bamboo
.venv/bin/jev-navi doctor
```

`configure-bamboo` 优先保存当前环境中的密钥，否则隐藏提示输入。保存到项目根目录 `.env`（权限 `0600`，已加入 `.gitignore`），后续所有 CLI 命令自动加载。已有环境变量优先；内容按字面值读取，不执行 shell。为避免覆盖，文件存在时保存命令会拒绝写入。密钥不进入实验配置或日志。该文件是本机明文文件，请勿分享。

也可仅使用当前终端环境变量：

```bash
# 用 read 隐藏输入并避免把密钥写进 shell 命令历史
read -r -s -p 'Bamboo API key: ' BAMBOO_API_KEY
export BAMBOO_API_KEY
export STARTLUX_ENDPOINT='https://api.bamboonode.dev/v1/decisions'

# d3-mini 使用独立密钥；inference.sh 页面标注按 GPU 时间收费
read -r -s -p 'Inference API key: ' INFERENCE_API_KEY
export INFERENCE_API_KEY
```

后续真实调用须获得新的阶段预算，并配置 `execution.mode=live`、独立 `phase_id`、`phase_limit` 和 `free_quota_confirmed=true`；现有历史及离线阶段均保持关闭。以下命令记录历史接口流程。静态契约检查先于导航实验；四种检查为当前图、按时序多图、逆序多图和空白图，总 HTTP 尝试上限 24，错误时停止：

```bash
.venv/bin/jev-navi contract --config configs/startlux.yaml \
  --images runs/habitat-smoke/0000/frames/0000.png \
           runs/habitat-smoke/0000/frames/0001.png \
           runs/habitat-smoke/0000/frames/0002.png \
  --instruction '走到前面的门口并停下' --output runs/startlux-contract
```

配置仍为 `free_quota_confirmed: false` 时命令会拒绝发出调用。HTTP 成功不证明图像被有效理解；需要用人工标注的导航图像与预期动作检查差异，契约结果默认保留 `image_grounding_verified: false`。

先进行六条短回合探索，再进行完整诊断。`startlux_pilot.yaml` 将导航尝试数限制在476以内，与一次静态契约检查合计最多500；这些是历史单次运行设置；当前所有真实请求还必须经过跨运行目录共享的阶段台账，不会因新目录重置阶段额度。六任务子集已按用户确认保存到 `configs/pilot_tasks.yaml`：城堡 visible-01/turn-01、卧室 visible-01/disambiguate-01、公寓 turn-01/disambiguate-01。每场景两条、每类型两条，保留原开发/留出划分；六条共享三个目标区域，不能视为六个独立目标。当前完整36任务清单也已就绪。

```bash
# 首先运行六任务 pilot（需在该配置中确认免费额度）
.venv/bin/jev-navi validate-suite configs/pilot_tasks.yaml
.venv/bin/jev-navi run --config configs/startlux_pilot.yaml \
  --suite configs/pilot_tasks.yaml --output runs/startlux-pilot

# 完整诊断在 pilot 分析后进行
.venv/bin/jev-navi validate-suite configs/local_tasks.yaml --diagnostic
.venv/bin/jev-navi run --config configs/startlux.yaml \
  --suite configs/local_tasks.yaml --output runs/startlux-diagnostic
```

认证、额度和不支持图像错误会终止实验并保存部分结果。每个请求超时 30 秒，可恢复网络/限流/服务端错误最多重试一次。总预算按 HTTP 尝试保守计数，包括重试与 d3 任务轮询；另记录模型提交数。网络超时后的重试可能再次提交已被服务端接受的任务，计费语义以提供商为准。

2026-10-27 已完成首次真实 StartLux 接口检查与六回合 pilot：接口无错误，导航成功0/6，发现区域外停止、循环和深度漏检。详见 [真实 pilot 诊断](docs/startlux_pilot_2026-10-27.md)。当前结果尚不支持进入完整评测。随后定位近处深度盲区，新增可选安全深度下倾30°；六回合复测碰撞25→0，成功仍0/6，详见 [安全深度诊断](docs/depth_diagnostic_2026-10-27.md)。 已追加 [循环终止与一次恢复反馈实验](docs/loop_experiment_2026-10-27.md)：固定阈值终止将请求132→71，反馈恢复128，三条件均0/6成功。

## 历史 local_v1 实验协议与结果

- 四动作：前进 0.25 米、左转 15°、右转 15°、停止。配置不能改变动作幅度而继续使用原提示词。
- 每次决策看到当前 RGB、最多两张旧 RGB、最近四条动作与执行反馈。图像 256×256、HFOV 90°，相机高 1.25 米。
- 深度仅供几何检查。目标坐标、GPS、导航图、目标距离与成功信息不进入策略。原始真值只供私有评测器和事后日志使用。
- 深度拒绝不移动、不等于 STOP；三次连续拒绝结束为 `stuck`。每回合最多 60 次决策，只有显式 STOP 且进入成功区域才算成功。
- 首轮 3 个场景共 36 回合：每场景三种任务各 4 条；开发场景与留出场景按场景族隔离。具体标注限制见 [任务标注](docs/task_annotations.md)。
- 条件为完整输入、无图像历史、固定的同场景指令置换；开发场景另做无深度拦截消融。指令置换不借用留出场景文本，也不修改评测目标。

输出包含 `suite_manifest.json`（冻结配置、任务、哈希、依赖与预算状态），每回合的 `manifest.json`、RGB/深度、`steps.jsonl`、`result.json`，以及 `episodes.csv`、`summary.json`、`report.md`、`replay.html` 和人工填写的 `failure_annotations.csv`。HTML 可离线打开。报告保留服务错误、原始提议、实际动作和拦截原因，按模型版本分组。

工程推进门槛为留出成功率 ≥60%，且完整输入相对打乱指令高 ≥15 个百分点。只有完成诊断、完整的 24 个留出回合配对、同模型版本且无相关服务故障时才判断门槛；部分结果显示证据不足。置信区间按场景族重采样，场景数量少时只作探索分析。

失败归因需要逐轨迹审核，程序不从“未成功”自动推断感知/方向/停止等原因。使用回放区分模型错误、几何误拦截、碰撞漏检与服务错误；同一场景下多个起点或语言变体不等于独立场景。

动作回放可在配置中使用 `policy: {kind: replay, path: runs/.../steps.jsonl}`。它读取记录的提议动作与原始响应，不发 API 调用，也不构成新的模型结果。

## 验证

```bash
OFFLINE=1 .venv/bin/python -m pytest -q
.venv/bin/ruff check src tests scripts
.venv/bin/ruff format --check src tests scripts
.venv/bin/python -m pip check
.habitat/bin/python -m pip check
```

测试覆盖深度单位/无效值与危险动作、停止语义、步数与调用预算、历史图像顺序、真值输入隔离、异步服务、概率解析、错误处理、诊断完整性与真实 Habitat 子进程往返。未安装原生环境时原生集成测试跳过，其余测试可独立运行。

研究目标、阶段边界和后续路线见 [研究协议](docs/research_protocol.md)。本项目没有导航训练或权重更新流程。

停止判断专项结果见[停止诊断](docs/stop_diagnostic_2026-10-27.md)：两次STOP后均有可行路线，静态历史对照发现执行反馈影响停止倾向；累计483次请求，尚无新的闭环成功证据。

开发任务清单与文件预检：见 [24条train任务冻结说明](docs/r2rce_task_preflight.md)。冻结定义不会授权真实调用或证明场景可导航。
