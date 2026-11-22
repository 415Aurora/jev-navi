# JEV-Navi：冻结多模态决策模型的局部导航实验

**Research portfolio summary:** Exploring frozen multimodal decision models for closed-loop, language-conditioned local navigation in Habitat. This is an exploratory research testbed, **not** a validated R2R-CE navigation benchmark.

**Research question:** Can a frozen multimodal decision model propose useful actions from language instructions, recent RGB observations and action feedback, when a depth-based safety layer independently checks motion feasibility?

**Implementation:** Action proposal adapters, metric-depth safety interception, isolated Habitat execution, reproducible task manifests, trajectory logging, replay and failure analysis. See [research protocol](docs/research_protocol.md) and [pilot report](docs/startlux_pilot_2026-10-27.md).

**Observed results (2026-10-27):** Initial six-episode model pilot: 0/6 successes. After the documented downward depth-safety adjustment, collisions in the diagnostic runs decreased from 25 to 0 while success remained 0/6. These small, task-specific comparisons do **not** establish general navigation improvement. See [depth diagnostic](docs/depth_diagnostic_2026-10-27.md) and [loop experiments](docs/loop_experiment_2026-10-27.md).

**Limitations:** No reported full R2R-CE score, zero-shot generalization claim or G1 deployment result. Scene count and trial count are limited. Reproduction of live-model trials requires provider access; synthetic/replay smoke tests should be reported separately.

---


本项目实现语言目标 → RGB 与短历史 → 冻结模型动作提议 → 米制深度拦截 → Habitat 单步执行的闭环。首阶段研究局部语义导航；完整 VLN、R2R-CE 和 G1/SONIC 迁移留待后续验证。

**证据边界：** 脚本策略、动作回放和合成环境仅用于软件验证，不视为模型导航结果。2026-10-27 已记录真实 StartLux API pilot（6 回合成功 0/6），但这不构成完整 VLN-CE benchmark 或有效导航策略的证据。见下文诊断记录。

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

## 真实模型接口

已实现 StartLux/Bamboo 与 inference.sh d3-mini 两种适配器。输入图像为 PNG data URI，严格按旧到新排列。模型选择取四种动作概率的最大值，保存完整原始响应；不把概率解释为导航安全置信度。

已核对 [Bamboo 官方 StartLux 文档](https://bamboonode.dev/docs/ai-model/decisions/startlux)（2026-10-27）：模型为 `startlux-decision-27b-autothink`，同步接口为 `https://api.bamboonode.dev/v1/decisions`，`/v1/systemone` 是兼容别名。当前测试期免费，零余额新账号无需充值；API Key 必须拥有 `default` 分组和该模型的访问权限。免费政策可能变化，运行前核对控制台。适配器显式使用 `think_mode=never`、`allow_none=false`，每张图像在 state 中有按时间排列的 `<image>` 标记。

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

确认可使用免费额度后，复制相应 YAML 并将 `free_quota_confirmed` 改为 `true`。静态契约检查先于导航实验；四种检查为当前图、按时序多图、逆序多图和空白图，总 HTTP 尝试上限 24，错误时停止：

```bash
.venv/bin/jev-navi contract --config configs/startlux.yaml \
  --images runs/habitat-smoke/0000/frames/0000.png \
           runs/habitat-smoke/0000/frames/0001.png \
           runs/habitat-smoke/0000/frames/0002.png \
  --instruction '走到前面的门口并停下' --output runs/startlux-contract
```

配置仍为 `free_quota_confirmed: false` 时命令会拒绝发出调用。HTTP 成功不证明图像被有效理解；需要用人工标注的导航图像与预期动作检查差异，契约结果默认保留 `image_grounding_verified: false`。

先进行六条短回合探索，再进行完整诊断。`startlux_pilot.yaml` 将导航尝试数限制在476以内，与一次静态契约检查合计最多500；更换供应商时另行配置并记录预算，不自动累计不同运行目录的费用。六任务子集已按用户确认保存到 `configs/pilot_tasks.yaml`：城堡 visible-01/turn-01、卧室 visible-01/disambiguate-01、公寓 turn-01/disambiguate-01。每场景两条、每类型两条，保留原开发/留出划分；六条共享三个目标区域，不能视为六个独立目标。当前完整36任务清单也已就绪。

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

## 实验协议与结果

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
.venv/bin/python -m pytest -q
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.habitat/bin/python -m pip check
```

测试覆盖深度单位/无效值与危险动作、停止语义、步数与调用预算、历史图像顺序、真值输入隔离、异步服务、概率解析、错误处理、诊断完整性与真实 Habitat 子进程往返。未安装原生环境时原生集成测试跳过，其余测试可独立运行。

研究目标、阶段边界和后续路线见 [研究协议](docs/research_protocol.md)。本项目没有导航训练或权重更新流程。

停止判断专项结果见[停止诊断](docs/stop_diagnostic_2026-10-27.md)：两次STOP后均有可行路线，静态历史对照发现执行反馈影响停止倾向；累计483次请求，尚无新的闭环成功证据。
