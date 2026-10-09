# 在线方法拆分与行为回归

日期：2026-10-09（北京时间）。范围：本地代码结构重构，方便分别阅读和修改
Vanilla、Dense、Fixed、Adaptive；本轮没有新 GPU 实验或性能结论。

## 改动

- 四种方法分别位于 `src/online_methods/vanilla.py`、`dense.py`、
  `fixed.py`、`adaptive.py`，附中文说明；阅读入口为该目录的 README。
- `online_protocol.py` 保留共同数据类型、置信度/D 停止规则、历史和成本
  更新，继续时委托相应调度函数。log/random/backoff 也各有独立文件。
- `online_engine.py` 接入独立 Vanilla 配置检查；主 token 循环、KV 后端、
  最终补答未改变。原公共配置和 Decision 记录结构保留。
- 静态源码清单覆盖新增 9 个方法模块及清单本身，运行/验收/回放门禁继承
  这些身份。模块缺失或哈希变化会被拒绝，不豁免旧 GPU 验收。

## 验证

本机已有隔离环境：`runs/online-validation/venv/bin/python`；
Torch 2.9.1、Transformers 4.51.3，CUDA 不可用。完整测试命令：

```bash
runs/online-validation/venv/bin/python -m unittest discover -s tests -v
```

**完整测试不是全绿：304 个测试方法中 293 通过、7 出错、4 跳过。**
7 个出错方法产生 8 条 error（一个方法有两个报错子用例），均来自旧教学测试
`test_upstream_diagnostic.py` 解析学习注释版
`upstream/CoDE-Stop/method_codestop.py:146` 的缩进错误。4 个跳过用例
需要本机缺少的 `math-verify[antlr4_13_2]==0.9.0`。

为区分已有问题与重构回归，将重构前 HEAD `0359ab0` 的原教学适配器和
测试文件导出到独立临时目录，只链接现有注释版上游，同一解释器得到完全相同
的 7 个出错方法/8 条 error。这两个文件的工作区内容与 HEAD 相同；运行前后
上游 15 个 Python 文件 SHA-256 均未变。学习注释未修改，错误未跳过或隐藏。
本次方法、公共引擎、后端及验收相关测试通过。

重构前保存的协议和引擎作为独立对照：

| 对照 | 结果 |
| --- | --- |
| a08/a09 的 50 个请求文件 | SHA-256 均保持 |
| 保存的 118 次探测决策 | 3,068 个顶层字段与新旧实现逐字段相同 |
| 46 配置、477 个合成场景 | 41,997 个 Decision 完全相同，随机状态一致 |
| Adaptive 公式/真正稀疏分支 | 覆盖 2,952 / 2,916 次合成决策 |
| 238 个完整引擎场景 | 169 正常返回、69 相同错误 partial；输出及后端动作相同 |
| 新增永久测试 | 方法行为 6 项、源码清单 5 项，另扩充分析器身份检查 |

这些计数是工程检查和历史记录重放，不是新增独立题目、GPU 请求或正确率样本。
CPU 对照不能验证重构后的 GPU 时延。

本地忽略目录 `runs/method-refactor-20261009/` 保存旧源码、独立对照脚本、
`audit.json`、`full-tests.log` 以及 `upstream-baseline/baseline-summary.json`。
独立行为审计 `audit.json` 的 SHA-256：
`18a076637ec62fa76fb0272685efdfafdd5de69a5324186c972d9d4b395f96e1`。

旧实验数据、输出及 release 均保留。当前源码身份已变化；后续若部署本版，
应重新完成对应有界 GPU 验收，不能把 a08/a09 的旧运行视为新版本的 GPU 验证。
