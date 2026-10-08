# 云端迁移交接：CoDE-Stop / When to Probe

更新：2026-10-08，北京时间。用途：供新对话或云端工作区接手研究。路径均相对于 CoDE 仓库根目录，执行命令前先确认当前位置。本文是当前迁移总入口；旧交接中的执行状态按其记录时间理解，不覆盖本文的更新。

**当前接续点：用户已完成一题教学 GPU 诊断和两题开发诊断。q001 已确认完整正确 boxed 值被旧提取方式遗漏；下一步接入明确答案边界与有界在线入口，再做单题 GPU 验收。调度纯逻辑、独立数学判分及共同在线组件已完成各自的本地检查；生产运行入口、真实 GPU 联调验收和论文主实验仍待完成。**

本文合并了 `docs/CONVERSATION_HANDOFF_20261008.md` 的执行记录，取代本文件原 2026-10-07 状态快照。最近已成功登录服务器，取回 q001/q002 等 33 个文件并逐项核对 SHA-256，又用原服务器 tokenizer 独立解码 CoDE 补答。教学题等未取回部分仍以用户回传日志为依据。详细交接可作为补充携带，本文可独立恢复任务背景。

用户已明确要求继续实验，本地随后新增独立诊断、判分、调度协议及在线推理组件。本文件只交接当前状态；未创建新云端任务、搬迁服务器或启动新 GPU 实验。云端工作区不自动拥有 GPU、模型缓存、服务器访问权或本地文件。

## 0. 迁移时先做这三件事

1. **把本文交给新对话。** 第 10 节有可直接复制的接手指令；研究目标、预算口径和当前问题均已保留。
2. **补齐工作文件。** 克隆仓库只能恢复已提交内容。按第 6 节另外携带未跟踪稿件、脚本及尚未提交的代码；接收后逐项检查文件和版本，不能把“已上传本文”当作“已迁移项目”。
3. **单独确认 GPU 访问和原始记录。** 先复用原服务器的模型与冻结数据。本文不携带认证信息；当前本机已能连接并完成只读检查；新云端仍须独立确认访问方式和已有任务，再启动新的推理。

建议阅读顺序：本节 → 第 2 节的真实进度 → 第 6 节文件清单 → 第 7 节第一步。第 3–5 节是研究方案和预算背景，可按需回查。所有完成状态以保存的文件、测试结果或真实运行证据为准。

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
| 当前优先问题 | q001 原因已查明，完整区间仍含重复 think 标记 | 冻结一致的边界处理，补齐有界在线入口 |
| 稀疏调度收益 | 两题 CoDE 均在前三次内停止 | 尚无默认前三次密集之后的稀疏机会 |
| 保存结果诊断、数学判分、调度纯逻辑 | 本轮新增，CPU 检查通过 | 待实际原始记录及 GPU 集成验收 |
| 共同在线推理组件 | 已保存并提交，本地状态机与 tiny Qwen3 检查通过 | 不当作 GPU 验收通过 |
| 正式运行入口、论文主表 | 入口尚缺、GPU 验收及主实验待做 | 保留 `TBD` |
| 当前 GPU | 单卡 RTX 4090，24564 MiB；检查时无 GPU 计算进程 | 租价和余额未核验；32K 容量及真实速度待测 |

### 已有 GPU 证据摘要

- **教学题** `37×43−29×41=402`：base 上限 2048 tokens；Vanilla 补答正确。DEER/CoDE 各一次 probe，confidence=0.78515625、未出现 `</think>`，均未早停并返回 Vanilla。验证了实际探测和回退路径。
- **q001** `math/train/algebra/566`，标准答案 2008：base 2442 tokens、65.7289 秒；Vanilla/DEER 回答正确。CoDE 第二次 probe confidence=0.9375、ended=true，超过当时 ramp 阈值 0.925，于位置 989 早停。补答用满 30 tokens；独立解码确认先生成完整 `\boxed{2008}`，之后又生成 `</think>` 并开始解释。旧 `rsplit` 遗漏了此前答案；严格判分仍因重复标记记为 needs_review，原记录保留。
- **q002** `math/train/geometry/428`，标准答案 997：base 用满 8192 tokens、215.5112 秒。DEER 五次 probe 均未 ended，未早停。CoDE 第三次退化分数 2.5050895895837346>2，于位置 5331 早停。Vanilla/DEER 输出 2996、CoDE 输出 2992，均错误，不能归为单纯格式问题。

置信早停和退化早停均有真实执行记录。当前 DEER/CoDE 基于 Vanilla 文本回放，`cp_cache=false`，每次 probe 重复预填充；阶段耗时不含基础生成。停止位置不能直接除以 base token 数计算正式节约率。停止后未执行的 probe 也不是完整密集采集。

另有历史 AIME Vanilla pilot：RTX 5090、Qwen3-4B BF16，10 题各生成一次，8 对 2 错，132,356 completion tokens，累计 2,943.931 秒，见稿件 `README_CN.md`。本轮未重读历史归档；其缺少逐步 probe／退化记录，不能证明当前服务器速度或正式准确率。

### 执行位置与访问边界

用户实际部署目录名是 **`codestop-gpu-prep-20261008`**，使用其中 `.venv`；不是后提供的 `gpu-auto` 包。连接信息通过用户已有安全配置提供，不写入可提交的摘要。

用户重启后已建立有效 SSH 控制连接，助手成功复用并登录新主机；私有详细交接第 2 节保存当前连接命令。已直接核验 RTX 4090、依赖、原项目和保存记录。此前认证失败是历史状态。新云端需要其自身的服务器访问方式，本机控制 socket 不能当作云端凭证携带。

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
- 本地 Math-Verify 0.9.0 独立判分已通过抽取、比较方向、异常与硬超时测试；已对 q001/q002 CoDE 保存的受控答案区间运行：q001 为边界 needs_review，q002 为 incorrect；正式在线结果接入仍待做。原版 API 判分属于另一协议，不混表。
- 按题进行配对分析和 bootstrap，两个种子属于同题聚类。开发阶段 2pp 筛选条件不等于已证明准确率无损。
- 离线反事实诊断可复用记录，但不作为在线加速证据。

## 5. 已讨论的硬件与预算

### 用户条件与授权状态

用户曾给出 RTX 5090、预算 500–1,000 元、连续 2–3 天；后来表示 A100/H100、多卡均可考虑，主要希望四天内完成。当前已现场确认单卡 RTX 4090 24GB；租价与余额尚未核验，不能把下面 A100 价格套到这台机器。以下是此前讨论的历史预算情景，本轮未刷新报价。

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
| `src/online_contract.py`、`src/online_engine.py`、`tests/test_online_engine.py` | 已保存的新共同在线接口与状态机；与 `online_protocol.py` 的最新修改一起携带，不能只取接口文件或只取旧提交 |
| `src/torch_online_backend.py`、`tests/test_torch_online_backend.py` | 已保存的 Torch 后端与 CPU 小模型测试；尚未真实 CUDA/BF16 验收，不在旧基线提交中 |
| 新增 `run_pilot_diagnostics.py`、`prepare_gpu_data.py`、`setup_gpu_server.sh`、`configure_gpu_remote.sh` | 均位于 scripts/，尚未跟踪；已有环境不自动重跑安装 |
| `requirements-gpu-prep.txt`、`docs/GPU_PREP_COMMANDS.md` | 准备依赖与说明，尚未跟踪 |
| `tests/test_pilot_diagnostics.py`、`tests/test_prepare_gpu_data.py` | 新增 CPU 工程检查，尚未跟踪 |
| 服务器 `data/benchmarks/` 的 JSONL、manifest.json、sources.lock.json | 已取回 manifest、source lock 和 pilot20；其余 JSONL 仍在原服务器 |
| 服务器 `runs/gpu-smoke-001/`、`runs/pilot20-diagnostic-8192/` | 已取回 pilot 顶层及 q001/q002 配置、完整 stage 和日志；教学题原始文件尚未取回 |
| 本地上游学习补丁、历史 AIME pilot 归档 | 保留原始证据及注释，与干净执行源码分开 |

**仅 clone GitHub 不会得到未跟踪的文稿、新入口和服务器结果。** 原始输出通过私有存储传递，不直接批量提交 Git。模型权重与 `.venv` 通常留在原 GPU 服务器，更换主机则按固定 revision 重建。密码、密钥、令牌、个人账户配置不进入迁移包。同步 `sources/` 参考材料只读。

迁移副本按以下四层检查；本文件没有代为打包或上传这些材料：

1. **已提交代码层**：确认实际 Git commit，并取得固定上游 Git object。协议测试会从 `upstream/CoDE-Stop` 读取固定提交的函数，只有当前源码而没有该 Git object 也不够。
2. **本地增量层**：保留需要的 tracked diff，以及表中明确列出的未跟踪稿件、脚本、配置、测试与在线组件。上游学习注释单独保留，执行用干净固定源码。不要整目录复制 `.vscode/`、全部缓存或机器配置。为实际携带的每个文件生成 SHA-256 清单；下面的几个 hash 只是关键身份，不是完整迁移包清单。
3. **服务器证据层**：携带 `data/benchmarks/` 的 JSONL、manifest/source lock；pilot 顶层 manifest、configs；q001/q002 全部 stage JSON、environment、console/events/launch 记录；以及服务器实际入口脚本的 hash。另保留精确 tokenizer revision 或其本地文件，便于不加载模型直接解码。已取回部分位于 `runs/remote-evidence-20261008/`，共 33 文件及 hash 清单；两份独立解码、新判分报告和当前环境清单也在该目录，均被 Git 忽略，须私下携带。
4. **新环境层**：按原记录重建解释器和依赖；不要把 Mac 的虚拟环境复制到 Linux。模型和数据已有缓存则校验后复用。旧 environment 内的绝对 snapshot 路径在新主机上可能失效，应显式提供实际 tokenizer 目录。

开始本次迁移更新时：主分支 main，HEAD `0ba16d843edb12ddb90aad230f940ce452ce5d8e`（`feat: add saved-answer audit grading and probe protocol checks`）；本次 `git ls-remote` 也核验远端 main 指向该提交。它包含保存答案检查器、判分组件及最初的纯逻辑调度协议。`6e5803f` 是此前交接文档提交。不能再沿用详细交接里的 `8d71ba7` 作为最新代码状态。

在线组件及禁用早停的密集采集扩展已另存为提交 **`3d520c6da2d8e926a33a247e004a9e54837b8ee0`**（`feat: add locally validated online inference components`），包含 contract、engine、Torch backend、protocol 修改及两份新增测试。接手时确认该提交确实已取到；推送与抓取成功须分别核验。本文的更新另用文档提交保存，不属于旧基线 `0ba16d8`。

已有教学 summary、上游学习文件及其他未跟踪内容均保留，禁止用清理或重置命令覆盖。**新代码入库不等于未跟踪稿件或服务器输出已迁移。**

初始部署包位于 `runs/gpu-prep-delivery/codestop-gpu-prep-20261008.tar.gz`。pilot 脚本随后单独上传，不在初始包内；只恢复该包不够。

本轮本地 SHA-256（证明文件身份，不代表实验验收）：

```text
ae1283d1301a58c371573643849ca7d021e2847c5a46fabc073dd5f97b2e57c1  manuscript/coling2027/main.tex
20f3e7281e212ef670cc8c3658210d796ddda40e07ae23d9835c5dd013554ec7  manuscript/coling2027/EXPERIMENT_PLAN.md
1af066dd294d104e9c80a25134cd7b2a0c0f43a0d9c70b2e8e4919c1567b17e9  manuscript/coling2027/references.bib
3529b14747bd36c127769908b103847e05eca4608544b2d552112735528310a3  scripts/run_pilot_diagnostics.py
7afcff48850f01dcda8685f9dc345fea438a62140d7795176ac0618aa23d5a6d  scripts/prepare_gpu_data.py
ef4ff6465802eb808885c9d5720d189b8cfbec7d2dbba40905982a674e190005  src/upstream_diagnostic.py
fbaf0f5ee39dc5420f8744c8fb6a1175b2cc2eb1758b8dd8dc2fa12bb3b610de  docs/CONVERSATION_HANDOFF_20261008.md
2510fd8581dd4d398302c5ccdf0c6e1cf27b0da81164192cb5967e4cc2c49efa  runs/gpu-prep-delivery/codestop-gpu-prep-20261008.tar.gz
c87156b2dfdb3ce12d069e676b6963329fefc734ca81a3f77b6d23a2b6ad3b11  src/online_contract.py
785fb930aa50847dde0d3b2237e4b14c1fdba4df8e078aeec3b63597dad11830  src/online_engine.py
ca5ada5c48d6d0d712ce2a4f3042e6dd6ed47dd9d961105e8dce915346978355  src/online_protocol.py
6aa6b639802afa98267c24d735e60e85a8f737cd2f63debdf040b60024e161e6  src/torch_online_backend.py
b6bc991f2c382f1962a8581c013f0de6ef598768aec8682375e84929712926cc  tests/test_online_engine.py
4ea5e2f33e51f5d8d1b8b30bd0b3bbf3681ed8a37fe4db0b3aecd2045e0919c9  tests/test_torch_online_backend.py
```

## 7. 云端接手后的准确第一步

### 本轮已新增的独立组件

| 文件 | 已验收范围 | 未完成范围 |
| --- | --- | --- |
| `scripts/inspect_saved_finalization.py` | 8 项 CPU 测试；真实 q001/q002 CoDE 的原 tokenizer 解码及 token 一致性已通过 | 其他实际输出按需核查 |
| `src/math_grading.py`、`scripts/grade_math_answers.py` | 22 项 CPU/实际 Math-Verify 集成测试；硬超时、抽取、完整分母、独立报告 | 实际答案区间输入及正式数据判分 |
| `src/online_protocol.py` | 25 项 CPU 测试；固定上游 D/ramp 对照、六种日程、随机流、无效观测及成本 EMA | 模型生成循环、KV 分叉、BF16 对齐和 GPU 计时 |
| `src/online_contract.py`、`src/online_engine.py` | 新增 16 项 CPU 状态机测试；包含停止／继续、答案边界、token 计数、预算与异常 | 与真实 Torch 后端联调、真实 GPU 验收 |
| `src/torch_online_backend.py` | 13 项固定版本 tiny Qwen3 CPU 测试；真实 DynamicCache 分支、私有 RNG、固定上游函数对照与采样顺序 | Qwen3-4B CUDA/BF16 数值、显存、完整引擎联调及真实耗时 |

配套测试为 `tests/test_inspect_saved_finalization.py`、`test_math_grading.py`、`test_online_protocol.py`；判分契约见 `docs/MATH_GRADING_PROTOCOL.md`。这些新增文件应一同迁移。旧 core/adapter/pilot/数据准备脚本均未修改，不改变原 pilot 的身份。

最近针对当前在线接口及状态机，执行 `python3 -m unittest discover -s tests -p 'test_online_*.py' -v`，41 项通过（协议 25＋状态机 16），覆盖本次 disabled-stopping 扩展。它不运行真实 Torch 后端测试。更早的完整测试在新增三项协议回归前为 87 项全过，不把两次范围不同的检查拼成一个虚构的全套通过数。测试中的临时仓库推送日志不代表真实项目已同步；实际远端引用已另核验。

另用本地 `runs/online-validation/venv/bin/python`、torch 2.9.1 和 transformers 4.51.3 重跑 `test_torch_online_backend.py`，13 项通过；没有下载 Qwen3-4B 权重。这是在 Mac CPU 上的小模型检查。该虚拟环境被忽略，不随 Git 迁移；新环境须按指定版本重建，真实 GPU 用原服务器环境另验。

### 共同在线组件的接续工作

`src/online_contract.py` 定义统一模型版本、标记和后端接口；`src/online_engine.py` 已实现生成、探测、提前退出及最终答案区间。`src/torch_online_backend.py` 已保存真实模型、KV 分支和采样实现，生产构造要求本地固定 revision、CUDA/BF16 及指定版本，不自行下载模型；CPU 测试使用显式标注 synthetic 的 tiny 随机模型。组件尚未完成真实 GPU 联调。新组件与旧回放 core 分开，不能覆盖被冻结的 pilot 实现。

需要保留的实现约定：

- 主生成先采样，再决定是否接纳 token 到 KV。遇到 `Wait` 时，以不含该 pending token 的前缀 probe；继续时接纳同一个 token，不重采样。
- probe 使用独立 KV 副本，不能改变主 KV、logits 和主随机流；随机日程另有自己的随机流。
- 自然 EOS 直接结束；思考阶段早停或耗尽预算时才注入共享 final prefix。答案阶段耗尽预算只续写，不重复关闭思考。
- 最终答案按保存的 token 边界提取，保留注入的 `\boxed` 前缀；不能按全文最后一个 `</think>` 重新切分。
- `dense_collect_no_stop` 密集记录停止谓词但继续生成。控制器仅允许 dense 日程关闭停止；`would_stop` 与实际 `should_stop` 分开记录。这属于采集成本，不是主表早停速度。
- CPU 合成小模型测试只验证接口与工程性质；原模型 BF16 数值、实际 CUDA KV 行为和墙钟必须在 GPU 上验收。

**`scripts/run_online_diagnostic.py` 尚不存在。** 云端接手应先检查组件及测试现状，再实现有界入口：冻结数据身份和代码哈希、按题与 rollout 派生独立种子、单卡单请求、显式 token／题数上限、新结果目录和锁、逐题原子保存、失败部分记录、原始输出与完整计时。不能提供不存在的“一键主实验”命令。首次只用一题和明确短预算作工程烟测，后续另验 32K 配置。

fixed/log/random 在有效但未结束的 probe 后保持预设日程；只有 adaptive/backoff 使用 incomplete 回退。所有家族共享 invalid/有效历史不足三条的密集回退。自然提前停止仍可发生在第三次之前。

### A. 核验已有文件和运行身份

确认可以访问稿件、代码和 q001/q002 原始记录。真实推理前核验实际主机、GPU、磁盘、运行进程与环境，避免重复启动任务。最近 nvidia-smi 返回 GPU 计算进程列表为空；这仅代表检查时状态，启动前仍须复查。

现有记录为 Python 3.12、torch 2.9.1+cu128、transformers 4.51.3；准备依赖另固定 accelerate 1.12.0、nltk 3.9.2、huggingface-hub 0.36.0、datasets 3.6.0、math-verify 0.9.0。完整环境以每题 `environment.json` 为准。不要先升级依赖、重跑安装脚本或覆盖 pilot。

数据已冻结：calibration80、另外 selection120、其子集 pilot20、math500、固定 analysis100。复用 JSONL、manifest 和 source lock，不重新抽样。来源是 `EleutherAI/hendrycks_math` train 和 `HuggingFaceH4/MATH-500` test，具体 revision 从远端锁文件读取。

独立判分环境还须满足 `antlr4-python3-runtime==4.13.2`、`latex2sympy2_extended==1.11.0`、`sympy==1.14.0`、`mpmath==1.3.0`；版本不符时判分 CLI 会拒绝运行。完整输入、抽取、超时与分母协议见 `docs/MATH_GRADING_PROTOCOL.md`。测试被跳过不算集成验收通过。

历史下载问题已解决：pip 改用官方 PyPI，权重下载用 `HF_HUB_DISABLE_XET=1` 避开 Xet 401。已有缓存优先复用，不每次接手都重装／重下。

### B. q001 只读解码已完成，保留复核方式

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

已确认新增内容先完成 `{2008}`，随后再次生成 `</think>`，旧 `rsplit` 确实跳过此前答案。q001 的答案不缺失，无需延长补答寻找正确值。核查报告见 `results/pilot20-finalization-audit.md`；这不表示旧提取代码已经修改或正式判分已经接受此异常输出。

- 已复用已有输出另存严格判分：q001 needs_review，q002 incorrect；原 grading 未改动。正确 boxed 值的边界审查结论单列。
- 当前无需对 q001 做 30→128 token 重生成。先明确如何对全部方法一致处理重复 think 标记，并在主测试前冻结；不得只为这个样本改规则。
- 接入、冻结并验证 Math-Verify。教学 grade 不覆盖一般 MATH 表达式；needs_review 不能静默当错，两题也不能支撑正式准确率。

答案边界核实后，另存逐请求 `final_answers.jsonl`（唯一 `id`、干净 `gold`、隔离的 `answer_text`、来源 hash 与边界说明），在装有冻结判分依赖的环境运行 `scripts/grade_math_answers.py`。须显式给出计划请求数，输出为新文件；不得回写旧 pilot 的 grading 或依靠 gold 修补输出。

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
| 1 | q001 原因核查已完成；携带证据并明确异常边界处理 | 原始 hash、token 解码、单列审查结论与严格判分 |
| 2 | 处理输出／判分，按固定顺序扩10–20题 | 第三次后仍继续比例、候选数、probe数、失败类别 |
| 3 | 将已测纯协议接入共同在线 GPU 后端和对照 | 候选一致、KV/RNG隔离、概率与固定参考对齐、完整计时 |
| 4 | 正式32K配置小样本验收 | 短中长轨迹、显存、平均及长尾耗时、实测费用 |
| 5 | 冻结校准、选参和判分 | 强fixed、重校准、rescale、简单退避；测试前冻结 |
| 6 | 在预算内完成主测试及消融 | 完整分母、逐题日志、配对统计与失败处理 |
| 7 | 据证据更新现有论文 | 论点、结果表、配置、引用一致，未做部分如实标注 |

共同在线组件首次 GPU 验收分成四个有界任务，均写入新目录：

| 验收 | 首轮范围 | 通过条件 |
| --- | --- | --- |
| probe 数值 | 最多 3 个已保存前缀，每个最多 21 个 probe token | 同前缀和固定上游逐 token、原始 BF16 概率、置信度、ended 标记对齐；预先约定容差 |
| KV／RNG 隔离 | 同一前缀，无 probe／插入 1 次／插入 2 次，各最多续写 64 tokens | 主 KV、logits、随机流不被探测改变，主续写逐 token 一致 |
| 答案区间接线 | 思考中补答、答案中预算耗尽、自然 EOS 三种路径 | 注入前缀与生成区间可追溯，不使用最后 think 标记重新切分，不额外救回 EOS |
| 在线计时与容量 | 1 题、Vanilla/dense/fixed，先限定 1024 主 tokens；32K 容量另验 | 全程在线、预热单列、计时与token分项齐全；目标最长 KV 下另查 context 和峰值显存 |

这些检查通过仅表示可以进入 10–20 题正式配置测速；不能据此声称方法加速或准确率保持。短预算 smoke 也不等于 32K 容量通过。失败时保存第一次差异，不静默换精度、预算或实现来拼表。

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
> 已完成教学题和两题开发诊断。q001已独立解码，确认原输出存在正确boxed=2008，旧最后think切片遗漏了它；q002的2992对标准997仍错。先读results/pilot20-finalization-audit.md，保留旧记录与新判分的边界区别，补齐有界在线入口并做单题GPU验收。不重复下载模型、重跑教学题或直接开完整矩阵。
>
> 随后从已保存的online_contract、online_engine和online_protocol接续，检查torch_online_backend的实际开发状态，补齐有界运行入口；不要把组件或CPU测试当作GPU验收。完成KV/RNG隔离、正式判分和32K小样本验收，用实测耗时更新预算。沿用已明确的执行授权；云端缺少服务器访问时先推进不依赖GPU的工作，并指出准确缺项，不宣称已接管服务器。
>
> 用中文说明第一条关键链路，只将可追溯真实结果写入论文，分别报告修改、测试、提交、推送和GPU验证。继续现有main.tex，不另起替代稿。
