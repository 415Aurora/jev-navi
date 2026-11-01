# MP3D 开发场景接入

用户于2026-10-29确认已获得MP3D下载许可；2026-11-01完成四个冻结train场景的数据接入。该声明不替代许可条款，数据不随Git仓库分发。

## 下载范围与来源

场景：759xd9YjKW5、GdvgFV5R1Z5、SN83YJsR3w2、aayBHfsNo7d；不更改已冻结的24条任务定义。
每场景保留glb、navmesh、house和semantic.ply，共16个文件，解压大小1,042,298,354字节。

官方脚本：https://kaldir.vc.cit.tum.de/matterport/download_mp.py
脚本SHA256：37bc39c84c72c1eafeb96f9290db0410217894b3858ee992cbaafb3a265c09ec。
官方脚本TASK_FILES中的habitat指向 https://kaldir.vc.cit.tum.de/matterport/v1/tasks/mp3d_habitat.zip 。
服务器声明大小16,085,306,031字节，支持HTTP Range。当前磁盘不足以稳妥保留整个包及所有解压数据，因此只获取四个场景的压缩成员，合计约746MB。

原脚本为Python2，且main在任务包下载后仍会提示继续全量下载。本次保留原脚本不修改，以现有Python3.9/lib2to3解释官方文件映射，不调用main；实际传输由curl读取官方ZIP的指定字节范围。这里没有宣称原脚本本身支持按house筛选Habitat包。
先读取ZIP尾部目录，再按选中成员的本地头偏移与压缩长度获取范围，写入临时稀疏文件；zipfile只读取选中成员，逐文件CRC校验，通过后由.part改名安装。稀疏文件不是完整可用的官方发布包，完成后已移除。
所有数据均保存在忽略目录；没有下载1.3TB原始发布，也没有下载其他场景主体。

原脚本、实际获取与验证脚本、范围清单、逐文件CRC和SHA256收据位于远端 `data/mp3d-download-source/`。
逐文件身份另记录于 `environment/mp3d-dev-assets.json`。本地哈希提供复现/变更检测，不是发布方签名；未对未下载的整个ZIP计算哈希或全包CRC。

## 验收与边界

文件预检重新导入原始标注并核对Git冻结清单；仍采用r2rce_v13_dev协议和现有Habitat0.3.3移植环境，不称为官方历史环境复现。
实际24/24条任务通过初始化与连通性检查；16个已安装文件的SHA256全部复核通过。原生检查逐条重置任务、读取224×224 RGB与256×256深度，验证起点在现有navmesh可导航且目标测地距离有限。报告在 `data/mp3d-download-source/native-readiness.json`。
该检查不证明策略能走完整路线，不改变安全层，不进行模型推理、训练或调参；没有用val_unseen做实验。现成navmesh的生成参数未在本阶段重新认证，不能仅凭配置声明agent半径就宣称原始navmesh与该半径严格一致。

真实模型HTTP为0。历史483/500和offline0预算保持；原生检查脚本验证台账字节哈希前后相同。导航源码、任务定义和main未改动。数据接入后248项测试、Ruff及57文件格式检查通过。

## 后续复查

在远端项目根目录：

```bash
OFFLINE=1 .venv/bin/jev-navi check-r2rce \
  --suite data/r2rce-source/frozen-train-dev24-v1/suite.json \
  --selection configs/r2rce_train_dev24.json --require-ndtw
OFFLINE=1 .venv/bin/python data/mp3d-download-source/check_native.py
```

check命令本身始终只报告文件级geometry_verified=false；原生检查结论单独保存，不能混为同一证据。
重复下载前核对收据和已有文件，当前一次性获取脚本拒绝覆盖已安装文件，不应直接重跑。
真实导航实验仍需模型部署、固定实际版本与推理设置，以及明确批准的新调用预算。
