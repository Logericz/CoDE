# 四种方法从这里读

本目录把“是否探测、下一次何时探测”按方法拆开。四种方法共用同一生成
引擎、同一模型后端；三个 CoDE 方法也共用停止规则。这样修改 Adaptive
的调度公式时，可以单独看它的文件，比较时又能保持其余环节一致。

## 先看四个文件

| 方法 | 文件及函数 | 继续推理时的行为 |
| --- | --- | --- |
| Vanilla | [vanilla.py](vanilla.py) / `create_controller` | 返回 `None`，不启用探测和早停；共享引擎正常生成 |
| Dense CoDE | [dense.py](dense.py) / `choose_next` | 每个候选点都探测：`h_next = 1` |
| Fixed CoDE | [fixed.py](fixed.py) / `choose_next` | 有效预热后按 `fixed_interval` 探测，本次开发实验设为 4 |
| Adaptive CoDE | [adaptive.py](adaptive.py) / `choose_next` | 先检查保护条件，再结合信号间隔和成本间隔 |

建议阅读顺序：Dense → Fixed → Adaptive → Vanilla 与共享生成循环。
Vanilla 文件很短，因为正常生成本来就由共同引擎负责。

**新增实际方法候选（2026-10-09）：** [guarded.py](guarded.py) 的 `choose_next`
允许数值有效但未完成的probe有条件跳过一个候选点。它是新增策略，原四方法保持
为对照。设计路线、公式与假设见 [Guarded设计](../../docs/GUARDED_SCHEDULER.md)。

`h_next` 的单位是候选点：4 表示下一次在当前候选点序号加 4 处探测，
中间跳过 3 个候选点。它不是 4 个 token 或 4 秒。当前候选点是在推理
阶段采样到单 token 的 `Wait`，探测前缀不含这个尚未接受的 token。

## 一次请求怎样经过这些文件

1. [run_online_development.py](../../scripts/run_online_development.py) 选择四个实验配置；
   [run_online_diagnostic.py](../../scripts/run_online_diagnostic.py) 的
   `configurations()` 将名称转换为方法和调度参数。
2. [online_engine.py](../online_engine.py) 的 `run_request()` 建立控制器。
   Vanilla 调用 `vanilla.create_controller()` 得到 `None`；
   CoDE 建立 `ProtocolController`，由 [__init__.py](__init__.py) 选择对应调度函数。
3. 共同引擎生成 token。遇到需要探测的候选点时，
   [torch_online_backend.py](../torch_online_backend.py) 的 `probe()` 在临时 KV 分支上生成试答。
4. [online_protocol.py](../online_protocol.py) 的 `observe()` 更新真实观测历史和成本，
   计算阈值、D 分数并检查停止条件。
5. 若停止，回到引擎执行最终补答；若继续，调用对应方法的 `choose_next()`，
   保存 `next_candidate_j = candidate_j + h_next`，然后继续主生成。

自然 EOS、自然思考结束和预算耗尽仍按共享引擎的既有优先顺序处理。
`dense_collect_no_stop` 是诊断采集模式：记录“本来会停止”但继续采集；
它使用 Dense 调度，不是第五种在线加速方法。

## Adaptive 中最值得修改和观察的地方

`adaptive.py` 按执行顺序分成三段，并配有中文说明：

1. **保护条件**：无效观测、未满 3 次有效观测、试答不完整、计时异常、
   信号不可用或置信度下降时，下次仍逐点探测。
2. **信号间隔**：根据距停止边界的余量 `margin` 和信号变化率 `activity`
   计算 `h_signal`。
3. **成本间隔**：用探测成本相对于每候选间隔推理成本的比值 `rho`，
   计算 `h_cost`，最后取 `min(h_signal, h_cost, h_max)`。

保护触发时不会执行后两段公式。停止判断先于保护条件，所以第 1 次探测
也可能提前停止；有效但未完成的试答仍可能通过共同的 D 规则触发停止。
Fixed 对有效但未完成的试答保持预热后的固定节奏，这个差异保留原协议。

[common.py](common.py) 定义字段冻结的输入上下文 `ScheduleContext` 和输出
`ScheduleResult`；随机调度会正常消耗其中独立 RNG 的状态。
这里的 `previous` 是上次有效观测，成本间隔来自上次实际
探测，两者不能混用。任何方法都不能拿跳过点或未来点的置信度更新历史。

## 改什么，应去哪个文件

| 修改目标 | 位置 | 影响范围 |
| --- | --- | --- |
| 自适应保护条件、间隔公式 | `adaptive.py` | Adaptive |
| 新候选的风险保护、1/2间隔决策 | `guarded.py` | Guarded，CLI为`codestop-guarded` |
| 固定间隔的调度逻辑 | `fixed.py` | Fixed |
| 固定间隔的实验取值 | `run_online_development.py` 中的 `configurations(LABELS, 4)` | 该实验配置 |
| 参数默认值和允许范围 | `online_protocol.py` 的 `ScheduleConfig` | 使用相关参数的配置 |
| 置信度阈值、D 分数、早停条件 | `online_protocol.py` | 所有使用共同规则的方法 |
| 试答生成、KV 分支、置信度计算 | `torch_online_backend.py` | 所有探测方法 |
| 主生成、自然结束、最终补答 | `online_engine.py` | 所有方法 |

`logarithmic.py`、`random_schedule.py`、`backoff.py` 是预声明的其他对照；
也已拆开，避免再次把调度逻辑塞回一个大分支。

## 验证与实验版本

在仓库根目录、本地 Mac 运行：

```bash
python3 -m unittest discover -s tests -v
```

此前四方法拆分为保持行为的结构重构：原配置名称、公共调用接口、停止规则和记录字段
保持兼容；不改学习注释、冻结数据及旧实验输出。旧 GPU 记录仍对应旧代码版本。
新增方法模块已加入源码哈希清单，后续部署必须包含整个目录。

Guarded随后增加了实际调度行为：仅当停止余量超过保护带，且按近期信号变化率
推算两候选内不会进入保护带时，取h=2，否则h=1；不沿用旧成本min。
无效probe和预热不足仍逐点查询，共同停止规则先于调度运行。
它只接入有界单题CLI，原十题批处理的四配置清单保持冻结。

当前源码不能直接冒用旧 GPU 验收或 a08/a09 的源码身份。要用新版本生成
结果，应按运行手册重新完成对应有界 GPU 验收；旧验收门禁仍会拒绝不匹配的
源码。CPU 对照通过说明所检验的决策一致，不等于新版的 GPU 时延已经实测。

本次测试、旧教学注释的现有缩进问题及对照结果见
[重构验证记录](../../results/online-method-refactor-20261009.md)。
