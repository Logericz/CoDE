# 云端迁移交接：CoDE-Stop / When to Probe

更新：2026-10-08，北京时间。用途：供新对话或云端工作区接手研究。路径均相对于 CoDE 仓库根目录，执行命令前先确认当前位置。

**当前接续点：用户已完成一题教学 GPU 诊断和两题开发诊断。先核验 q001 的补答／答案提取，再决定是否扩到 10 题。调度纯逻辑与独立数学判分已新增并完成 CPU 检查；真实在线 GPU 后端和论文主实验仍待完成。**

本文合并了 `docs/CONVERSATION_HANDOFF_20261008.md` 的执行记录，取代本文件原 2026-10-07 状态快照。GPU 数字来自详细交接保存的用户回传日志；本轮没有取得远端完整原始文件，不能称为助手已直接复核。详细交接可作为补充携带，本文可独立恢复任务背景。

迁移文档完成后，用户明确要求继续实验；本轮随后新增独立诊断、判分和调度协议组件，未创建新云端任务、搬迁服务器或启动新 GPU 实验。云端工作区不自动拥有 GPU、模型缓存、服务器访问权或本地文件。

## 1. 已确定的方向与协作方式

- 目标：COLING 2027；现有题目 **When to Probe: Separating Observation and Stopping in Reasoning Models**。
- 用户已决定继续 CoDE-Stop 探测调度与早停方向。TokenSkip 组合已讨论，暂不作为主线；也未决定以 API 替代 GPU 主实验。
- 核心问题：减少中间答案探测，既改变开销，也改变停止规则看到的历史和退出机会，如何影响正确率、停止位置和端到端耗时？
- 自适应调度仍为候选方法。先与调优 fixed、阈值重校准、rescale 和简单退避比较，不能预设有创新或收益。
- 用户希望约 4 天内完成实验验证，这是目标，不是已测定工期。正式矩阵尚依赖后端实现、判分与性能验收。
- 使用中文；说明命令在哪运行、目的和成功条件。用户希望理解第一条关键链路，再让助手承担重复验证、汇总与文档。
- 分别报告本地修改、测试、提交、推送、远端核验和 GPU 运行；只有可追溯真实结果才能写进论文。保留学习注释、固定数据及已有输出。

## 2. 已完成工作、当前问题与访问状态

| 项目 | 当前记录 | 接手处理 |
| --- | --- | --- |
| 草稿、实验方案、参考文献、审查报告 | 本地存在，文稿尚未纳入 Git | 单独携带，继续原稿 |
| 依赖、模型、MATH 数据准备 | 用户报告服务器已完成 | 检查已有清单，优先复用 |
| 教学 GPU 链路 | `runs/gpu-smoke-001/` 四阶段完成 | 不重复同一验收 |
| 开发诊断 | `runs/pilot20-diagnostic-8192/` 的 q001、q002 完成，failed=0 | 阶段完成不等于答案正确 |
| 当前优先问题 | q001 CoDE 补答／提取异常 | 按第 7 节只读解码 |
| 稀疏调度收益 | 两题 CoDE 均在前三次内停止 | 尚无默认前三次密集之后的稀疏机会 |
| 保存结果诊断、数学判分、调度纯逻辑 | 本轮新增，CPU 检查通过 | 待实际原始记录及 GPU 集成验收 |
| 正式在线 GPU 后端、论文主表 | 待实现、待测 | 保留 `TBD` |
| 当前 GPU 型号、卡数、余额与任务进程 | 本轮未直接取得 | 从实际主机及环境记录核验 |

### 已有 GPU 证据摘要

- **教学题** `37×43−29×41=402`：base 上限 2048 tokens；Vanilla 补答正确。DEER/CoDE 各一次 probe，confidence=0.78515625、未出现 `</think>`，均未早停并返回 Vanilla。验证了实际探测和回退路径。
- **q001** `math/train/algebra/566`，标准答案 2008：base 2442 tokens、65.7289 秒；Vanilla/DEER 回答正确。CoDE 第二次 probe confidence=0.9375、ended=true，超过当时 ramp 阈值 0.925，于位置 989 早停。补答用满 30 tokens；按最后一个 `</think>` 提取的片段不完整，完整输出是否已有答案仍待核验。
- **q002** `math/train/geometry/428`，标准答案 997：base 用满 8192 tokens、215.5112 秒。DEER 五次 probe 均未 ended，未早停。CoDE 第三次退化分数 2.5050895895837346>2，于位置 5331 早停。Vanilla/DEER 输出 2996、CoDE 输出 2992，均错误，不能归为单纯格式问题。

置信早停和退化早停均有真实执行记录。当前 DEER/CoDE 基于 Vanilla 文本回放，`cp_cache=false`，每次 probe 重复预填充；阶段耗时不含基础生成。停止位置不能直接除以 base token 数计算正式节约率。停止后未执行的 probe 也不是完整密集采集。

另有历史 AIME Vanilla pilot：RTX 5090、Qwen3-4B BF16，10 题各生成一次，8 对 2 错，132,356 completion tokens，累计 2,943.931 秒，见稿件 `README_CN.md`。本轮未重读历史归档；其缺少逐步 probe／退化记录，不能证明当前服务器速度或正式准确率。

### 执行位置与访问边界

用户实际部署目录名是 **`codestop-gpu-prep-20261008`**，使用其中 `.venv`；不是后提供的 `gpu-auto` 包。连接信息通过用户已有安全配置提供，不写入可提交的摘要。

本轮已有 SSH 尝试返回 `Permission denied (publickey,password)`：连接已进入认证阶段，助手未登录成功，未取得 GPU 或进程信息。旧详细交接中的 DNS 失败是此前状态，不能概括本轮失败原因。云端需要其自身的服务器访问方式，不在聊天或 Git 中传递密码／私钥。

本地上游有学习修改：`inference.py`、`method_deer.py`、`method_prompts.py`、`models.py`。严格校验曾报 `Upstream source differs from pinned version: models.py`。已部署 prep 包从固定 Git objects 导出干净参考副本；不要清除本地注释、改固定哈希，或为解决本地差异覆盖远端冻结代码。

## 3. 研究定义与不可混淆的边界

参考论文：*Early Stopping for Large Reasoning Models via Confidence Dynamics*，CoDE-Stop，arXiv:2604.04930v2。

- 论文：<https://arxiv.org/html/2604.04930v2>
- 作者仓库：<https://github.com/sudoparsa/CoDE-Stop>
- 固定上游提交：`b5081e7c2abe23bb1d19649421cc13522fee7c50`
- 用户项目仓库：<https://github.com/Logericz/CoDE>。本次同步时 GitHub 提示原地址 `Ericzhy0716/CoDE` 已迁移到此处；旧文档仍可能保留原地址。
- 主模型：`Qwen/Qwen3-4B`
- 模型与 tokenizer revision：`1cfa9a7208912126459214e8b04321603b3df60c`

### 已有分析

固定公开代码的默认 `use_log=True` 退化分数可以化简为按 token 位置加权的置信下降事件计数。在相同置信轨迹、有效有限数值、严格递增正时间戳、相同首尾及共同终点等条件下，稀疏子序列的分数不大于密集序列。少于三次观测时，按该实现置零。

该命题不直接推出实际停止更早/更晚、准确率提高或在线加速；也不能直接推广到论文概率域表达式和代码 `use_log=False` 变体。三个定义必须分别标注。

要区分三种影响：

1. 跳过检查点，失去一次退出机会。
2. 跳过观测，改变历史退化统计。
3. 用“实际探测次数”代替“原候选位置”推进阈值，可能改变阈值时钟。

**默认前三点密集、ramp=2 时，两个时钟在后续均已饱和，时钟因子是零效应控制。** 无 warm-up 的额外诊断才用于暴露可能的时钟变化，不能把它冒充默认主方法的现象。

### 候选方法

有界自适应探测：利用已经观察到的停止余量、变化率和实际 probe 成本决定下次查询间隔；有效历史不足三条、置信下降或异常时回退密集；间隔设上限。完整公式、参数与伪代码见 `main.tex` 和 `EXPERIMENT_PLAN.md`。

需要回答的是它是否优于调优后的简单策略。若固定策略重校准解释了全部收益，应收缩方法主张，保留有证据的机制分析或负结果。

“少探测”“考虑探测成本”已有相关研究，不能直接声称首次。LearnStop 是直接外部比较对象；更广的方法级主张还涉及 ASAG、BLADE、ReCo 等，需按实际比较覆盖限制结论。

## 4. 当前实验方案

### 数据与规模

| 用途 | 规模 | 规则 |
| --- | --- | --- |
| 工程检查 | 已有教学题与已暴露 pilot，必要时少量新开发题 | 不作为未见测试集 |
| 停止阈值校准 | MATH 训练划分 80 题 | 固定题号、来源和哈希 |
| 日程选择 | MATH 训练划分另一组 120 题 | 与校准和测试按题隔离 |
| 主测试 | 完整 MATH500 × 2 个生成种子 | 参数冻结后运行 |
| 完整密集记录 | 开发 200 条＋主测试预先固定 100 题的首个种子 | 继续至自然结束或预算终点，单列采集成本 |
| 核心机制消融 | 主测试固定 100 题，首个种子，先做 3 个消融 | 子集结论不替代完整主表 |
| AIME 扩展 | 60 题中未用于历史 pilot 的 50 题，需先核验题号 | 已暴露 10 题单列，不混成未见结果 |

主表的七个基础对照：Vanilla、DEER rule、密集 CoDE rule、调优 fixed、调优 log、匹配开发 probe 数的 random、主候选 adaptive；另加 `fixed-rescale`、`fixed-recalibrate`。简单双倍退避也必须比较，可放完整主表或机制表并注明范围。

**最新预算采用更完整的 10 配置口径**，即把简单双倍退避也放到整个主测试：500 × 2 × 10 = 10,000 次在线请求。`README_CN.md` 中的 9,000 次只算九配置，不是新增一套相冲突的实验结果。

经济调参方案：五个可调日程家族各先离线评估九配置，再在线选两个，线上约 1,200 次；固定阈值重校准另有 1,080 次、缩放控制 120 次，并为三种基础方法各预留 120 次，共约 2,760 次。加 300 次核心消融，总计 **13,060 次在线请求**。默认完整九配置线上调参对应约 **17,260 次**。

上述数量尚不包括 300 条完整密集采集、缺失候选停止点的最终补答和失败重跑。离线筛选也可能需要真实 GPU 补答；不能直接拿 probe 猜答代替最终答案。

采用经济调参模式的前提：缓存有完整置信观测及必要分段计时；所有日程家族使用相同缩减规则；在打开主测试结果前冻结。若条件不满足，不能宣称已经实施该省钱模式。

### 正式运行约束

- Qwen3-4B thinking、BF16、单卡每次一请求；多卡用于独立任务分片。
- 主生成 temperature=0.6、top_p=0.95、top_k=20、min_p=0；正式最大新生成 token 为 32,768，另检查输入加输出的上下文容量。
- probe 贪心，保留固定实现最多 21 个生成 token 及原概率语义；原方案计划统一最终补答最多 30 token；当前补答异常须在正式冻结前处理，这项规格尚未验收。协议变化显式说明。
- 主推理与探测 KV 分支隔离，主推理与随机日程使用独立随机流；不能因多做 probe 而改变主推理随机序列。
- 请求计时覆盖 prefill、主推理、probe、缓存准备/恢复、最终答案及必要控制开销。记录准确率、平均耗时、probe 数、token、显存及失败分类。
- 计划使用本地 Math-Verify；准备依赖已列入 0.9.0，但正式抽取、比较方向、异常与超时尚未接入验收。原版 API 判分属于另一协议，不混表。
- 按题进行配对分析和 bootstrap，两个种子属于同题聚类。开发阶段 2pp 筛选条件不等于已证明准确率无损。
- 离线反事实诊断可复用记录，但不作为在线加速证据。

## 5. 已讨论的硬件与预算

### 用户条件与授权状态

用户曾给出 RTX 5090、预算 500–1,000 元、连续 2–3 天；后来表示 A100/H100、多卡均可考虑，主要希望四天内完成。已有服务器运行记录，但实际当前 GPU 型号、卡数、租价与余额尚待核验。以下是此前讨论的历史预算情景，本轮未刷新报价。

截图中的候选为 A100 PCIe 40GB，单卡 10 CPU 核、72GB 内存，系统盘 30GB、数据盘 50GB，标价 **¥3.28/卡时**；当时页面只有一张空闲卡。四卡供应与单价不能由这张截图保证。截图镜像为 PyTorch 2.8，不满足当前教学工具的 torch 2.9.1 强校验。所有价格、库存、驱动和磁盘在实际购买前刷新。

### GPU 规划

助手建议核心验证准备 **¥1,500–2,000**，后续补实验另留约 ¥1,000，整体资金池先按 ¥3,000 安排；先投入约 ¥100 做环境和正式配置小样本测速。**这些是建议，用户尚未明确授权按此金额开机或充值。** 沿用以后用户明确给出的执行授权，不重复索取已经给过的确认。

以 13,060 请求及 ¥3.28/卡时计算：

| 假设平均完整请求耗时 | 纯在线推理卡时 | 纯在线推理租金 | 含附加工作及储备的规划额 |
| --- | ---: | ---: | ---: |
| 60 秒 | 217.67 | ¥714 | ¥1,050–1,250 |
| 90 秒 | 326.50 | ¥1,071 | ¥1,500–1,700 |
| 120 秒 | 435.33 | ¥1,428 | ¥1,900–2,100 |

规划额另计附加工作/存储 ¥150–300，再留约 20% 储备。**耗时和附加金额均为情景，不是实测或完工保证。** 探测密度、重复前缀、长尾和故障偏高时需重估。

四卡连续 96 小时纯租金为 **¥1,259.52**，提供 384 卡时。若为其他工作留出 20% 卡时，13,060 次在线请求平均需约不超过 85 秒，才有机会装进四天；还需考虑分片不均和调参先后依赖。增加卡数主要缩短工期，不自动减少总卡时。

若资金上限坚持 ¥1,000，优先考虑一随机种子的缩小版研究，明确降低交付范围，而不是声称同样的完整矩阵必能完成。

### API 比较的结论

未决定转为 API。2026-10-07 查询 OpenRouter 原 `qwen/qwen3-4b` 的端点列表为空，不代表其他平台全部没有该模型。

此前查询的 DeepInfra Qwen3-14B 公共页为 **FP8、40,960 上下文**，标准价每百万输入 $0.12、输出 $0.24；不是当前 Qwen3-4B BF16 的等价服务。文档有 raw completions 与 logprobs，但具体模型的 thinking 前缀、特殊 token、试答概率语义、版本固定、32K 新生成及缓存行为未做付费实测。

对同样 13,060 个完整任务，假设平均主推理 5,000 token、题目/模板 500、每次试答 21、最终补答 30，前缀均长取主推理一半，在线中断和续写都重传前缀、无缓存优惠，以 $1=¥7.2 的预算汇率估算：

| 假设平均探测次数 | 输入/百万 token | 输出/百万 token | 纯 token 费 | 规划资金 |
| --- | ---: | ---: | ---: | ---: |
| 5 次 | 470.16 | 67.0631 | 约 ¥522 | ¥800–1,000 |
| 20 次 | 1,645.56 | 71.1770 | 约 ¥1,545 | ¥2,200–2,500 |

API 规划额大致留 50% 附加工作储备及 ¥50 杂费。5/20 次均不是当前方法的实测平均；各组需最终按真实 usage 分开计价。该表跨模型/精度，只用于成本量级，不证明 API 更便宜或能完成当前协议。

API 能做适配后的信号/服务成本研究，但网络、排队、重复 prefill 与供应商缓存属于该部署的成本。先完整生成再探测不能证明在线提前停止省时。此前建议核心实验继续用 GPU；若试 API，先给 ¥20–50 兼容性测试额度，额度本身仍不是已执行或已授权付款。

官方参考（报价和能力在使用前刷新）：

- <https://api.autodl.com/docs/price/>
- <https://openrouter.ai/api/v1/models/qwen/qwen3-4b/endpoints>
- <https://deepinfra.com/Qwen/Qwen3-14B>
- <https://docs.deepinfra.com/apis/completions>
- <https://docs.deepinfra.com/chat/log-probs>

## 6. 必须携带的文件与版本

仅上传本文能恢复背景，不能替代代码、稿件和原始实验记录。

| 材料 | 用途／迁移注意事项 |
| --- | --- |
| `docs/CLOUD_MIGRATION_HANDOFF.md` | 本文件，云端总入口 |
| `docs/CONVERSATION_HANDOFF_20261008.md` | 详细执行摘要，尚未跟踪，含机器路径，私下迁移 |
| `manuscript/coling2027/` | main.tex、EXPERIMENT_PLAN、references.bib、README、ACL 样式；尚未跟踪 |
| `ccfa-review-reports/when-to-probe-coling2027-review.md` | 稿件问题清单，尚未跟踪 |
| `AGENTS.md`、`UPSTREAM.json`、`requirements-diagnostic.txt`、`src/`、`configs/`、`scripts/`、`tests/` | 固定参考、工程入口、依赖和测试，完整迁移其相互依赖 |
| 新增 `run_pilot_diagnostics.py`、`prepare_gpu_data.py`、`setup_gpu_server.sh`、`configure_gpu_remote.sh` | 均位于 scripts/，尚未跟踪；已有环境不自动重跑安装 |
| `requirements-gpu-prep.txt`、`docs/GPU_PREP_COMMANDS.md` | 准备依赖与说明，尚未跟踪 |
| `tests/test_pilot_diagnostics.py`、`tests/test_prepare_gpu_data.py` | 新增 CPU 工程检查，尚未跟踪 |
| 服务器 `data/benchmarks/` 的 JSONL、manifest.json、sources.lock.json | 冻结划分、revision 和校验和，本轮未取回 |
| 服务器 `runs/gpu-smoke-001/`、`runs/pilot20-diagnostic-8192/` | 原始逐题记录、配置、环境、日志、token IDs，本轮未取回 |
| 本地上游学习补丁、历史 AIME pilot 归档 | 保留原始证据及注释，与干净执行源码分开 |

**仅 clone GitHub 不会得到未跟踪的文稿、新入口和服务器结果。** 原始输出通过私有存储传递，不直接批量提交 Git。模型权重与 `.venv` 通常留在原 GPU 服务器，更换主机则按固定 revision 重建。密码、密钥、令牌、个人账户配置不进入迁移包。同步 `sources/` 参考材料只读。

本轮更新本文前：主分支 main，HEAD `8d71ba76ba4c37479b11787ef47823c4ef4b1b84`。本地跟踪记录显示与 origin/main 一致，远端实时状态需单独核验。已有教学 summary、上游学习文件及其他未跟踪内容均保留。

初始部署包位于 `runs/gpu-prep-delivery/codestop-gpu-prep-20261008.tar.gz`。pilot 脚本随后单独上传，不在初始包内；只恢复该包不够。

本轮本地 SHA-256（证明文件身份，不代表实验验收）：

```text
ae1283d1301a58c371573643849ca7d021e2847c5a46fabc073dd5f97b2e57c1  manuscript/coling2027/main.tex
20f3e7281e212ef670cc8c3658210d796ddda40e07ae23d9835c5dd013554ec7  manuscript/coling2027/EXPERIMENT_PLAN.md
1af066dd294d104e9c80a25134cd7b2a0c0f43a0d9c70b2e8e4919c1567b17e9  manuscript/coling2027/references.bib
3529b14747bd36c127769908b103847e05eca4608544b2d552112735528310a3  scripts/run_pilot_diagnostics.py
7afcff48850f01dcda8685f9dc345fea438a62140d7795176ac0618aa23d5a6d  scripts/prepare_gpu_data.py
ef4ff6465802eb808885c9d5720d189b8cfbec7d2dbba40905982a674e190005  src/upstream_diagnostic.py
3c1e007a52e0e45feb723464764785fe915214d763af00b5cf7968ad449a6181  docs/CONVERSATION_HANDOFF_20261008.md
2510fd8581dd4d398302c5ccdf0c6e1cf27b0da81164192cb5967e4cc2c49efa  runs/gpu-prep-delivery/codestop-gpu-prep-20261008.tar.gz
```

## 7. 云端接手后的准确第一步

### 本轮已新增的独立组件

| 文件 | 已验收范围 | 未完成范围 |
| --- | --- | --- |
| `scripts/inspect_saved_finalization.py` | 8 项 CPU 测试；本地 tokenizer 解码、token 前后缀一致性、全部 thinking 边界 | 真实 q001/q002 解码尚未取得 |
| `src/math_grading.py`、`scripts/grade_math_answers.py` | 22 项 CPU/实际 Math-Verify 集成测试；硬超时、抽取、完整分母、独立报告 | 实际答案区间输入及正式数据判分 |
| `src/online_protocol.py` | 25 项 CPU 测试；固定上游 D/ramp 对照、六种日程、随机流、无效观测及成本 EMA | 模型生成循环、KV 分叉、BF16 对齐和 GPU 计时 |

配套测试为 `tests/test_inspect_saved_finalization.py`、`test_math_grading.py`、`test_online_protocol.py`；判分契约见 `docs/MATH_GRADING_PROTOCOL.md`。这些新增文件应一同迁移。旧 core/adapter/pilot/数据准备脚本均未修改，不改变原 pilot 的身份。

完整测试在新增三项协议回归前为 87 项全过；最后仅调整协议组件，对其重跑 25 项通过。均为工程检查，测试中的临时仓库推送日志不代表真实项目已同步。此前迁移摘要提交为 `6e5803f`，真实 GitHub 推送因连接失败未完成；后续提交与推送状态看实际 Git 记录。

fixed/log/random 在有效但未结束的 probe 后保持预设日程；只有 adaptive/backoff 使用 incomplete 回退。所有家族共享 invalid/有效历史不足三条的密集回退。自然提前停止仍可发生在第三次之前。

### A. 核验已有文件和运行身份

确认可以访问稿件、代码和 q001/q002 原始记录。真实推理前核验实际主机、GPU、磁盘、运行进程与环境，避免重复启动任务。本轮没有助手直接核验的远端活动进程清单。

现有记录为 Python 3.12、torch 2.9.1+cu128、transformers 4.51.3；准备依赖另固定 accelerate 1.12.0、nltk 3.9.2、huggingface-hub 0.36.0、datasets 3.6.0、math-verify 0.9.0。完整环境以每题 `environment.json` 为准。不要先升级依赖、重跑安装脚本或覆盖 pilot。

数据已冻结：calibration80、另外 selection120、其子集 pilot20、math500、固定 analysis100。复用 JSONL、manifest 和 source lock，不重新抽样。来源是 `EleutherAI/hendrycks_math` train 和 `HuggingFaceH4/MATH-500` test，具体 revision 从远端锁文件读取。

历史下载问题已解决：pip 改用官方 PyPI，权重下载用 `HF_HUB_DISABLE_XET=1` 避开 Xet 401。已有缓存优先复用，不每次接手都重装／重下。

### B. 只读解码 q001，不加载模型、不重新生成

新增独立工具可输出包含源文件哈希的完整 JSON，先将该工具上传到原项目而不覆盖任何已冻结脚本，再在原项目根目录执行：

```bash
.venv/bin/python scripts/inspect_saved_finalization.py \
  runs/pilot20-diagnostic-8192/items/q001/codestop.json
```

工具只向 stdout 输出，不修改原始记录。若迁移了 tokenizer，使用 `--tokenizer-path` 指定已存在的本地目录。以下为同一核查目的的简化逐段展示命令。

在实际 **gpu-prep 项目根目录**，使用原 `.venv` 执行。该命令只读取记录和本地 tokenizer。在新云端执行需先有记录及相应 tokenizer，不能直接使用旧环境 snapshot 的绝对路径。

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
from transformers import AutoTokenizer

p = Path("runs/pilot20-diagnostic-8192/items/q001")
env = json.loads((p / "environment.json").read_text(encoding="utf-8"))
r = json.loads((p / "codestop.json").read_text(encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(
    env["model"]["snapshot"], local_files_only=True, trust_remote_code=False,
)
call = r["generation_calls"][-1]
inp = call["input_token_ids"][0]
out = call["output_token_ids"][0]
new = call["generated_token_ids"][0]
response = r["method_output"]["response"]
print("输出是否保留完整输入前缀:", out[:len(inp)] == inp)
print("补答输入末尾:", tok.decode(inp[-120:], skip_special_tokens=False))
print("实际新增的全部 token:", tok.decode(new, skip_special_tokens=False))
print("保存回答末尾:", response[-1500:])
print("回答中的结束标记数量:", response.count("</think>"))
PY
```

判断新增内容是否再次生成 `</think>`，使 `rsplit` 跳过先前答案；或者补答确实未完成。目前两者均待核验，不能写成已发现并修复的 bug。

- 若只是提取问题，复用已有输出，另存有版本的判分结果，保留原始记录。
- 若确实补答未完成，另目录做仅改变补答长度的诊断，先评估 30→128 tokens，检查前 30 tokens 能否复现；不要同时改长度和采样。这项补答复用工具尚未实现／运行。
- 接入、冻结并验证 Math-Verify。教学 grade 不覆盖一般 MATH 表达式；needs_review 不能静默当错，两题也不能支撑正式准确率。

### C. 处理问题后再复用原目录续跑

pilot 固定全部 20 题、seed=42、上限8192、顺序与身份；默认 limit=2。**limit 不进入身份，可从 2 扩到 10/20 并复用完成项。** 脚本、配置、数据、源码或环境发生变化则需新目录，不能改校验值蒙混续跑。先比对远端脚本与 manifest，不先上传覆盖。

解决补答／提取问题并确认沿用原协议后，才使用：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/run_pilot_diagnostics.py run --limit 10
.venv/bin/python scripts/run_pilot_diagnostics.py inspect --limit 10
```

`inspect` 会重写 summary.json，不是严格只读；退出码0也可能有 incomplete，须核对 requested completed/failed/incomplete 与逐题记录。

现有入口仍为工程回放，上限8192；不能把 JSON 改为32768就当作正式在线后端。相关 CPU 测试本轮审计通过：pilot 6项、数据准备4项；这是工程证据，不是 GPU 或在线速度验收。

## 8. 后续任务及验收

| 顺序 | 工作 | 必须保留的证据 |
| --- | --- | --- |
| 1 | 补齐文件、核验 q001 补答 | 文件身份、完整 token 解码与答案边界 |
| 2 | 处理输出／判分，按固定顺序扩10–20题 | 第三次后仍继续比例、候选数、probe数、失败类别 |
| 3 | 将已测纯协议接入共同在线 GPU 后端和对照 | 候选一致、KV/RNG隔离、概率与固定参考对齐、完整计时 |
| 4 | 正式32K配置小样本验收 | 短中长轨迹、显存、平均及长尾耗时、实测费用 |
| 5 | 冻结校准、选参和判分 | 强fixed、重校准、rescale、简单退避；测试前冻结 |
| 6 | 在预算内完成主测试及消融 | 完整分母、逐题日志、配对统计与失败处理 |
| 7 | 据证据更新现有论文 | 论点、结果表、配置、引用一致，未做部分如实标注 |

若大多数题前三次就停止，默认调度可节省的探测有限，应重新评估主张，不只挑probe多的题。若强fixed解释了全部收益，收缩自适应优越性主张。

正式逐题记录至少包括：数据版本／划分／题号、模型和tokenizer revision、种子、代码和配置哈希、软硬件、原始答案及判分、停止原因、候选/token位置、置信度和退化分数、主推理/probe/补答token、端到端耗时、峰值显存。

当前脚本本身没有租金上限。扩量前落实剩余预算、停止时间和监控方式；沿用用户明确给过的授权，不重复询问。配置错配、数据重叠、KV/RNG污染、重复OOM、记录丢失或触及费用上限时停止受影响运行。

## 9. 文稿和规则的遗留事项

- 用户最新决定优先；执行事实以原始记录和当前代码为准。详细GPU回传见 `CONVERSATION_HANDOFF_20261008.md`，旧复现交接中的“从D1开始”“无GPU结果”已过时。
- `UPSTREAM.json` clean/experiments_started 等字段是历史状态，不能代替现场检查。
- main.tex 对 fixed 重校准范围的文字，与方案仅在选定间隔 `h*` 上重校准不同。本预算按后者；正式运行前统一规格。
- 当前诊断最终补答沿用上游采样，正式方案计划统一贪心，这是协议差异，需记录和验收。
- 文稿尚未吸收最新两题诊断；本轮未编辑或编译 main.tex，旧PDF与源码是否一致待写作时核验。
- COLING 2027 截止、ARR周期、页数和审稿服务规则在实际提交前查官方最新来源。此前日期是旧规划，本文件不确认当前官方规则。合作者／审稿服务人选尚未确认本次参与。
- 费用脚本 `estimate_paper_budget.py` / `estimate_api_budget.py` 仅做情景计算，不是运行入口。旧三方法共享轨迹预算不能代替13,060请求矩阵预算。
- 遵守用户最新要求及项目AGENTS.md，保留其他未提交工作，自有实现放src/scripts。

## 10. 可直接交给云端新对话的指令

> 请依据这份 CLOUD_MIGRATION_HANDOFF.md 接手我的 CoDE-Stop / When to Probe 项目，目标COLING 2027。继续探测调度与早停方向，不转TokenSkip，不默认用API替代主实验。
>
> 先清点你能访问的稿件、代码和服务器原始记录。仅clone得不到未跟踪的论文、新pilot脚本和远端结果。延续原稿和固定数据，不从零规划。
>
> 已完成教学题和两题开发诊断。先按第7节只读解码q001的CoDE补答，区分提取问题和实际截断；保留原始记录及冻结身份，不重复下载模型、重跑教学题或直接开完整矩阵。处理后再考虑续跑固定前10题。
>
> 随后推进共同在线后端、KV/RNG隔离、正式判分和32K小样本验收，用实测耗时更新预算。沿用已明确的执行授权；云端缺少服务器访问时先推进不依赖GPU的工作，并指出准确缺项，不宣称已接管服务器。
>
> 用中文说明第一条关键链路，只将可追溯真实结果写入论文，分别报告修改、测试、提交、推送和GPU验证。继续现有main.tex，不另起替代稿。
