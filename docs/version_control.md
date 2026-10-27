# 版本管理约定

## 仓库与分支

工作目录：/home/elwen/Projects/jev-navi。
GitHub 私有仓库：415Aurora/jev-navi，SSH origin：git@github.com:415Aurora/jev-navi.git。

- main 保存已审查、完成相应验证的版本。
- baseline-2026-10-27 标记首次纳入 Git 的项目现状，不代表导航能力通过。
- 后续开发分支使用 codex/ 前缀；下一阶段为 codex/r2rce-offline-hierarchy。
- 每个独立改动单独提交；提交说明交代目的及实际完成的验证。
- 合并前审查 diff 与检查结果，不把现存问题描述为已修复。

## 纳入与排除

跟踪源码、测试、脚本、配置、研究文档、依赖锁文件、环境复现清单和环境检查脚本。
不跟踪密钥、本地环境、工具缓存、第三方源码、场景数据、运行产物和环境生成日志。
vendor/ 中 Habitat 的独立 Git 仓库不作为子模块加入；版本由
environment/source-manifest.json 等清单记录。

实验运行前记录 git rev-parse HEAD、git status --porcelain、
运行配置哈希与输出目录；工作区有改动时保留差异或提交后再运行。
密钥仅保留在已忽略的本机文件或环境变量中。

## 日常操作

~~~bash
cd /home/elwen/Projects/jev-navi
git status
git switch -c codex/r2rce-offline-hierarchy
# 完成独立修改并运行相应验证后，明确选择要提交的文件
git diff
git add <files>
git diff --cached
git commit -m "feat: describe the verified change"
git push -u origin HEAD
~~~

## 首次基线执行记录

- 本阶段只建立版本管理，未修改导航行为、未调用模型 API。
- 按已批准计划在现有远程目录初始化 main，不另建工作树。
- 新增内容仅为忽略规则和本说明；原始源码、配置、文档及环境资料按初始化前哈希核对。
- 基线保留既有脚本格式问题；本提交不混入格式或功能修复。
- Git 验收以跟踪文件审查、忽略规则检查、原文件哈希、远端提交及标签一致性为准。
