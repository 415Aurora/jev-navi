# R2R-CE 数据接入预检

此工具只读取文件，不创建仿真环境/策略，不触碰调用台账或发送模型HTTP。
先用 prepare-r2rce 将官方格式导入，再检查准备清单：

~~~bash
OFFLINE=1 .venv/bin/jev-navi check-r2rce --suite runs/prepared-suite.json
OFFLINE=1 .venv/bin/jev-navi check-r2rce --suite runs/prepared-suite.json --require-ndtw
~~~

ready=true退出0；文件缺失、源哈希不符、缺目标、空任务或要求nDTW而缺dense GT时退出2。
schema错误同样返回2并输出error。每次重新检查实际场景文件，不信任导入时缓存的缺失列表。
原始episodes/GT需保留在清单记录的路径，迁移后应重新prepare，不能静默改写来源哈希。
--require-ndtw检查dense GT覆盖，不检查计算依赖；运行时仍需requirements-evaluation.lock中的fastdtw。

source.kind=official只是输入者的声明。declared_official_data_ready不证明许可、上游真实性、
人工标注正确性或navmesh兼容；geometry_verified和live_authorized始终false。
之后还需原生reset/可达性检查、冻结研究任务与独立模型预算。文件级ready不表示真实模型已获调用许可。

## 正式数据准备
- 采用R2R_VLNCE_v1-3；保留episodes来源与可选dense GT原文件及哈希。
- scenes-root应指向data/scene_datasets，场景相对路径保持mp3d/<scan>/<scan>.glb。
- 测试split无目标，不能在本地算SR/SPL；当前未实现官方test提交器。
- 保持train与val_unseen角色独立；先确定小规模train开发集，不自动获取全部场景。
- 远端当前约26GB空闲，按所选场景实际大小规划，不默认可容纳完整MP3D。
- 获取MP3D访问需要用户自行完成授权流程；项目不代签条款或自动获取受限场景。

官方资料：[VLN-CE格式与GT](https://jacobkrantz.github.io/vlnce/data)、
[MP3D访问流程](https://niessner.github.io/Matterport/)。
