# 有界共同在线入口与 GPU 验收

2026-10-09 代码导航：四种方法已拆到 [src/online_methods/](../src/online_methods/README.md)，
共同引擎与停止规则保持共享。部署时须包含新增目录及 `scripts/online_source_manifest.py`。
源码哈希清单已更新；下文旧验收是旧 release 的历史证据，不能当作重构版的 GPU 验收。
继续生成前重新建立当前源码的对应验收记录；不要改写旧 manifest 或冻结 release。

范围：一题、一个 rollout，默认顺序运行 Vanilla / dense CoDE / fixed CoDE。默认使用已暴露的 pilot20 q002 (`math/train/geometry/428`)；首次主预算 1024 tokens，每次最终补答最多 30 tokens，每个 probe 最多 21 tokens。此入口最多允许 8192 主 tokens，不是论文完整矩阵或 32K 容量验收入口。

新增可选配置为 `deer-dense`、`codestop-log`、`codestop-random`、`codestop-backoff`、`codestop-adaptive`、`dense-collect-no-stop`。默认配置不变；必须显式选择扩展配置。无早停密集采集记录 would-stop 后继续，不进入在线速度主表。随机日程使用独立派生种子，所有请求保存实际参数及 `config_hash`。probe 开始/结束为实时日志；`probe_decision` 标为 `post_request_summary`，在请求返回后输出。

`scripts/run_online_diagnostic.py` 验证冻结 manifest 的外部 SHA-256 锚点、全部数据文件、source lock、题号、问题哈希、开发/测试隔离及子集关系。不会抽样或下载数据/模型。运行目录必须不存在；同一物理 GPU 的 UUID 锁跨目录共享，发现已有计算进程即拒绝运行。

## 机制与协议

- 主流与日程种子按题号、rollout、master seed 和用途分别派生；三个对照共享主流种子。dense/fixed 不使用随机日程，因此实际 `seed_schedule=null`，另存保留的日程种子。
- 在 `Wait` 被采样后、接纳到主 KV 前探测。probe 克隆 KV，继续时接纳原来的同一个 token，停止时丢弃但计入生成成本。
- 协议 `common-online-validation-v2` / 引擎 `1.1.0` 识别 tokenizer 主 EOS 与 `model.generation_config.eos_token_id` 的并集。现场固定 Qwen3-4B 的两个 EOS 为 151645、151643；主生成、补答、probe 异常检查统一使用完整集合。
- 自然 EOS 立即终止；思考中早停/耗尽预算注入共同 final prefix。已经自然关闭思考后耗尽预算，只续写答案，不再关闭思考。
- 答案来源为保存的 token 区间，包含注入的 boxed 前缀；只移除区间末尾的一个已知 EOS，用 `skip_special_tokens=False` 解码。重复 think 原样保留，严格判分仍送复核。`answer_boundary_confirmed=false` 不进入数学比较。详见 [判分协议](MATH_GRADING_PROTOCOL.md)。
- 模型加载、16-token 预热请求和判分分开记录。请求计时包含 prefill、主生成、probe/KV、补答、解码、控制与进度日志开销；每次操作前后 CUDA 同步，适用于该测量后端，不能外推到未插桩服务。

## 执行前

在本机确认 SSH 通往已授权的当前服务器；连接命令只保留在私有交接。本机或别的云端 CPU 不是 GPU 验收环境。先检查远端主机、`nvidia-smi`、计算进程、磁盘、Python、Torch/CUDA/BF16、Transformers 与判分依赖。复用原 `.venv` 和固定 snapshot，不运行安装或下载脚本。

所有下面命令均在**远端独立代码部署目录**执行；原 prep 项目只读。先导出固定 Git object `method_codestop.py` 到新目录的 `upstream/CoDE-Stop/`，保留其 SHA-256 `d020fde481bf1ad2d4ee53ea43a3ae31560e05be1d6cf6f172489b54179cb1ff`，不要复制带学习注释的工作树来替换它。新代码应附逐文件 SHA-256 清单，上传后核验一致。

```bash
export CODESTOP_PREP_ROOT=/root/autodl-tmp/codestop-gpu-prep-20261008
export CODESTOP_MODEL_DIR="$CODESTOP_PREP_ROOT/data/model-cache/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

## 数值与隔离验收

先纯读取检查输入计划；不加载模型，也不生成结果目录：

```bash
"$CODESTOP_PREP_ROOT/.venv/bin/python" scripts/validate_online_gpu.py \
  --inspect-only \
  --evidence-root "$CODESTOP_PREP_ROOT/runs/pilot20-diagnostic-8192" \
  --upstream-source-dir upstream/CoDE-Stop
```

真实 GPU 检查最多三个已保存前缀（默认 q001:0、q001:1、q002:0），不重新生成完整解答。容差在运行前固定为 exact；对照相同 attention 实现、BF16、前缀与上游固定函数，比较逐 token、原始概率、confidence、ended。第一次差异保存后失败退出，不改容差、精度或 attention 来拼接通过结果。0/1/2 次插入 probe 的三次续写各最多 64 tokens；核对 KV、logits、主 RNG、全局 RNG 和主续写一致。状态哈希与参考运行耗时不是在线推理性能结果。

```bash
"$CODESTOP_PREP_ROOT/.venv/bin/python" -u scripts/validate_online_gpu.py \
  --model-dir "$CODESTOP_MODEL_DIR" \
  --evidence-root "$CODESTOP_PREP_ROOT/runs/pilot20-diagnostic-8192" \
  --upstream-source-dir upstream/CoDE-Stop \
  --attention-implementation eager --continuation-tokens 64 \
  --run-root runs/gpu-validation-001
```

## 一题在线烟测与独立判分

以下 manifest 锚点是 2026-10-08 现场读取的冻结文件身份，漂移时停止检查，不替换为新 hash 蒙混通过。此命令只选已暴露的 q002；不重跑 q001 寻找 2008。

```bash
"$CODESTOP_PREP_ROOT/.venv/bin/python" -u scripts/run_online_diagnostic.py \
  --data-dir "$CODESTOP_PREP_ROOT/data/benchmarks" \
  --data-manifest-sha256 ae236d14cfa3b5cfbf434163386fea878016355051313ccdbd11d0a309eebca7 \
  --model-dir "$CODESTOP_MODEL_DIR" \
  --sample-id math/train/geometry/428 \
  --max-new-tokens 1024 --master-seed 42 --rollout-id 0 \
  --configurations vanilla codestop-dense codestop-fixed \
  --fixed-interval 4 --attention-implementation eager \
  --run-root runs/online-q002-1024-001

"$CODESTOP_PREP_ROOT/.venv/bin/python" scripts/grade_math_answers.py \
  --input runs/online-q002-1024-001/final_answers.jsonl \
  --planned-count 3 \
  --output runs/online-q002-1024-001/math-grades.json
```

stdout 与 `events.jsonl` 同步给出模型加载、预热、每 64 主 tokens、probe 开始/结束和逐请求状态。每个请求原子保存结果与判分输入；失败保留部分 token、pending token、probe、计时、配置及环境，不执行后续配置。`summary.json` 分别报告执行、未执行、失败和源文件是否在运行中改变。判分在模型请求之外独立执行，输出必须是新文件。

运行后逐项确认：三个计划请求是否都已保存、源/数据身份一致、token 守恒、时间分项闭合、停止原因与答案边界、严格判分状态、峰值显存，以及实际覆盖了哪些生成路径。短预算没有出现的自然 EOS、答案阶段预算耗尽或早停分支不能算作 GPU 覆盖。单题耗时不证明日程收益，0 个 probe 也不能算 probe 在线接线已验收；32K 容量、长轨迹及主测试均需另行验收。

## 完整前缀 exact 失败后的独立定位

`scripts/diagnose_probe_parity.py` 只接受已保存的同前缀 exact 失败，核验原始记录、源码、模型、tokenizer、运行版本与数值协议一致。它复用原完整前缀结果 C，不改变原失败；同次模型加载只新增三条 probe：A 为逐 token KV＋新 probe，D 为同一份 A 的 KV＋固定原函数（真实 `deepcopy/cache_to_device`），B 为整段前缀 KV＋新 probe。每条最多21 tokens。接着独立做0/1/2 probe隔离，各最多64 tokens。

该工具需额外导出固定 `cache_utils.py`，SHA-256 为 `e4bcea4a8c9bb5f76a4e2cacf7070c4c302de4272645d974e1de008636f73d04`。部署到另一个新目录；不得覆盖已运行的release。它自动沿用失败运行的attention和seed，无放宽容差或切精度选项。

```bash
"$CODESTOP_PREP_ROOT/.venv/bin/python" -u scripts/diagnose_probe_parity.py \
  --model-dir "$CODESTOP_MODEL_DIR" \
  --evidence-root "$CODESTOP_PREP_ROOT/runs/pilot20-diagnostic-8192" \
  --failure-root /root/autodl-tmp/codestop-online-acceptance-20261008-a01/runs/gpu-validation-001 \
  --upstream-source-dir upstream/CoDE-Stop \
  --run-root runs/parity-diagnosis-001
```

退出码2表示诊断完成但严格比较仍有失败；退出码1表示运行中断。结果分开报告same-KV、bulk-prefix、完整前缀及隔离，不能把其中一项通过写成总GPU验收通过。定位到前缀构建路径也不自动证明是某个BF16舍入或kernel机理。最新实测状态见 `results/online-gpu-acceptance-20261008.md`。

## 五个保存前缀的同 KV 验收

新的验收契约见 [ONLINE_REFERENCE_CONTRACT.md](ONLINE_REFERENCE_CONTRACT.md)。使用新目录，原完整前缀 exact 失败保持原样。固定 q001 两点和 q002 三点；同 KV 新/原 probe 精确一致与隔离为当前门槛，完整前缀另作敏感性对照。停止历史按题分别维护，只将仍在思考阶段的 Wait 计为在线候选。

以下命令在**新的远端独立部署目录**运行。`--inspect-only` 先检查原始证据/identity/源码，随后移除该参数并提供模型及全新输出路径才执行 CUDA：

```bash
"$CODESTOP_PREP_ROOT/.venv/bin/python" -u scripts/validate_online_continuation.py \
  --evidence-root "$CODESTOP_PREP_ROOT/runs/pilot20-diagnostic-8192" \
  --identity-root /root/autodl-tmp/codestop-online-acceptance-20261008-a02/runs/parity-diagnosis-001 \
  --upstream-source-dir upstream/CoDE-Stop \
  --model-dir "$CODESTOP_MODEL_DIR" \
  --run-root runs/same-kv-five-prefixes-001
```

每个前缀最多三条 21-token probe 路径；另在 q002 首点插入 0/1/2 次 probe 后各续写最多 64 tokens。保存逐概率差异、默认停止判定及草稿阈值网格的离线敏感性。该网格只复用已测概率，不是重新调参或新增在线方法结果。源码、输入及身份在运行前后检查，失败或中断保留已完成证据。

## 合成题上的真实自然边界

`scripts/validate_online_boundaries.py` 使用固定问题 `Compute 1 + 1.`、seed=872，默认512主tokens，允许运行前明确指定1024或4096。先要求真实采样自然关闭思考并输出EOS；成功后，第二请求以 `closing_token_index+1` 为预算，第三请求以“首个使自然答案前缀解码后非空白的非控制、非EOS token的index+1”为预算。正文定位只跳过空白，不使用gold，要求后面仍有自然EOS，整段自然答案不得含额外控制标记。

两个重复请求都必须与第一条相应主采样前缀精确一致（token、概率、阶段及位置），不注入前缀，仅贪心续写最多30个答案tokens。这样分别验证自然关闭思考当下、以及非空答案正文已经开始后的预算分支。首次主预算固定在manifest中，不因未结束而自动增加或换seed。

```bash
"$CODESTOP_PREP_ROOT/.venv/bin/python" -u scripts/validate_online_boundaries.py \
  --model-dir "$CODESTOP_MODEL_DIR" \
  --run-root runs/natural-boundaries-4096-001 --max-main-tokens 4096
```

命令在新的远端独立部署目录执行。未覆盖时退出码2，分别保存已执行请求、已覆盖分支、未执行数量和总体状态；后续失败不能抹掉前面的真实覆盖，但代码身份变化会使证据不再满足前置门槛。它只验证真实模型执行分支，来源标为合成题，不进入数据集准确率或速度结果。旧a05固定512请求未自然结束的记录保持原样；4096是另行预声明的新运行。

## 合成 32K 缓存容量

`scripts/validate_online_capacity.py` 只用于 CUDA 显存/形状验收。它固定读取 a01 q002 Vanilla 的已保存 1024 个 thinking tokens（源码中锁定文件 SHA），循环填充 32768 tokens，默认每块 128；绝不把 32K 一次整段输入 eager attention。满长时执行真实 probe 并检查主 KV/logits/RNG 隔离，再注入 final prefix 和最多 30 个贪心补答 tokens。

命令必须指定 `--source-record`、新五点验收的 `--gate-summary`、`--natural-run-root`（可多次）、模型和新运行目录。默认要求实际在线早停、自然 EOS、自然关闭思考后的预算终止都有已核验记录；后两项可以来自上述合成题真实模型工具，但记录中分别标明来源，不能冒充数据集自然终止覆盖。仅开展独立容量诊断时可显式选择 `--independent-capacity-diagnostic`；缺少的自然分支仍写为未通过，不绕过同 KV 数值门槛。`--inspect-only` 只校验计划，不加载 GPU。

运行记录包括 source/gate/环境/代码身份，逐 1024 tokens 的 allocated/reserved/peak/free 显存与各层 KV 形状、满长 probe、补答及执行前后文件 hash。默认 900 秒墙钟预算在 CUDA 操作之间检查，不能抢占单个 kernel。任何错误/OOM 即保存失败并退出，不改精度、attention 或缩块重试。容量通过不能写成自然生成 32K、32K 数值等价、长轨迹准确率或在线速度通过。

本轮实际容量命令使用 `--independent-capacity-diagnostic`，因为自然 EOS 和自然关闭思考的预算分支未覆盖；没有把未覆盖的运行作为通过的前置证据。

```bash
"$CODESTOP_PREP_ROOT/.venv/bin/python" -u scripts/validate_online_capacity.py \
  --source-record /root/autodl-tmp/codestop-online-acceptance-20261008-a01/runs/online-q002-1024-001/vanilla.json \
  --gate-summary /root/autodl-tmp/codestop-online-continuation-20261008-a03/runs/same-kv-five-prefixes-001/summary.json \
  --natural-run-root /root/autodl-tmp/codestop-online-continuation-20261008-a03/runs/online-q002-8192-001 \
  --model-dir "$CODESTOP_MODEL_DIR" \
  --run-root runs/capacity-32k-001 --chunk-size 128 \
  --independent-capacity-diagnostic
```

## 固定q002的32K自然生成验收入口

`scripts/run_online_long_context.py` 是独立的单题工程入口，固定已暴露q002、Vanilla、master seed42/rollout0，只允许明确指定32768主tokens，最终补答仍限30。原8192入口上限不变。它要求五点same-KV及64-token隔离门槛、实际早停/自然EOS/答案阶段预算三个分支、原32K合成容量、冻结数据、当前核心源码与执行环境均匹配；缺少任何前置证据即拒绝启动，无独立诊断绕过选项。

所有命令在新远端独立部署目录执行。先加 `--inspect-only` 做纯读取检查；移除该参数才加载GPU。容量验收的7-token probe和有限显存余量原样披露，不保证未观察到的最长probe路径。

```bash
"$CODESTOP_PREP_ROOT/.venv/bin/python" -u scripts/run_online_long_context.py \
  --data-dir "$CODESTOP_PREP_ROOT/data/benchmarks" \
  --data-manifest-sha256 ae236d14cfa3b5cfbf434163386fea878016355051313ccdbd11d0a309eebca7 \
  --gate-summary /root/autodl-tmp/codestop-online-continuation-20261008-a03/runs/same-kv-five-prefixes-001/summary.json \
  --capacity-root /root/autodl-tmp/codestop-online-continuation-20261008-a05/runs/capacity-32k-001 \
  --natural-run-root /root/autodl-tmp/codestop-online-continuation-20261008-a03/runs/online-q002-8192-001 \
  --natural-run-root /root/autodl-tmp/codestop-online-continuation-20261008-a06/runs/natural-boundaries-4096-001 \
  --max-new-tokens 32768 --model-dir "$CODESTOP_MODEL_DIR" \
  --run-root runs/online-q002-32768-001
```

加载、16-token预热和判分单列；请求实时记录每64个主tokens，保存完整答案、token/probability、分项计时、显存和前后身份校验。OOM或异常仅保存该次失败，不自动重试/降预算/切精度；普通RequestError保留部分生成证据，无法获得引擎部分记录的中断诚实标记缺失并保留已有events。完成后用原严格判分器单独评分，再与旧a03相同seed的主轨迹前缀核对。

32768是允许的上限，不保证每次自然生成都达到该长度；单题成功不证明主表准确率或任何调度收益。


## 固定10题开发成本试跑与tmux托管

`scripts/run_online_development.py` 复用已完成的same-KV、真实分支、合成容量和a07长轨迹证据。按冻结pilot20原顺序排除 `math/train/algebra/566` 与 `math/train/geometry/428`，取余下前10题，记录0-based来源行和原ID，编号dev01–dev10。历史q001/q002不是原JSONL的前两行，不按行号猜身份。

每题固定Vanilla、dense CoDE、fixed(interval4)、adaptive既有默认配置，四配置按题循环换序，共40请求。主预算32768、probe上限21、最终补答上限30，master42/rollout0；同题共享主随机种子，每请求重建私有采样流、KV与控制器。此轮用于开发成本和执行诊断，不做参数搜索。旧异常复核包继续pending，新输出仍按原严格协议判分。

入口无题目、配置、种子或预算覆盖选项。`--inspect-only` 仅验证文件；`--preflight-tokenizer-only` 使用已缓存的固定tokenizer，禁用Torch/TensorFlow/Flax，不加载权重。实际运行再次检查全部题目的 `prompt + 32768 + reserve <= model context`。旧容量只测过q002的151-token prompt；新题更长部分是这次真实试跑的范围，不按长度筛题、缩预算或假称已有容量实测。

在新的**GPU服务器release目录**中执行，前置路径沿用上一节相应真实结果：

```bash
"$CODESTOP_PREP_ROOT/.venv/bin/python" scripts/run_online_development.py \
  --data-dir "$CODESTOP_PREP_ROOT/data/benchmarks" \
  --data-manifest-sha256 ae236d14cfa3b5cfbf434163386fea878016355051313ccdbd11d0a309eebca7 \
  --gate-summary "$SAME_KV_RUN/summary.json" \
  --capacity-root "$CAPACITY_RUN" \
  --natural-run-root "$Q002_8192_RUN" \
  --natural-run-root "$NATURAL_BOUNDARIES_RUN" \
  --long-context-root "$Q002_LONG_RUN" \
  --model-dir "$CODESTOP_MODEL_DIR" \
  --preflight-tokenizer-only
```

总墙钟上限7200秒，涵盖模型加载、预热、请求和中间I/O；每请求上限1800秒，与总时限同时生效。计算/解码操作前检查，正在执行的CUDA kernel不可抢占。失败、OOM或超时终止矩阵并保存partial，无自动重试。每请求独立保存 `requests/devNN/configuration/request.json` 和 `final-answer.json`，再append+fsync到 `answers.jsonl`；结束时另存完整 `final_answers.jsonl` 与40请求分母的summary。未执行请求单列。

长任务使用**服务器tmux**，不能在Mac上的tmux里只包一个SSH命令。当前私有release中的 `supervise_development.py`、`launch-arguments.json`、`predeclared-plan.json` 均纳入部署hash：服务器依次执行生成、独立严格判分、完成回执，console/log和结果均落远端磁盘。托管脚本不会覆盖旧目录或自动重跑；即使生成非零退出，若已发布答案文件，也对已执行行按 `--planned-count 40` 判分。

```bash
# 在GPU服务器上；SESSION与RELEASE使用本次已核验的新会话/目录。
tmux new-session -d -s "$SESSION" -c "$RELEASE" \
  "$CODESTOP_PREP_ROOT/.venv/bin/python" -u "$RELEASE/supervise_development.py"
tmux ls
tmux attach -t "$SESSION"
# 查看后按Ctrl-b，再按d，退出查看而保持任务运行。
```

Mac休眠、关闭终端或SSH断开后，服务器tmux继续运行；服务器自身停机或重启不在该保障范围。重新连接后读 `console.log`、`runs/development10-001/events.jsonl` 与 `completion.json`。completion分别记录生成和判分退出码，并在收尾核查部署身份；summary中的生成完成不代替后续判分成功。

报告所有已执行请求的耗时、失败partial和未执行覆盖，另列完整四配置配对。若时间上限只留下部分完整题，不能用幸存题均值代表全部10题或主测试；10题顺序也不保证覆盖全部题型和长短轨迹。判分未决、人工复核pending与执行失败分别统计，保持原始结果不回写。

## 同10题的完整密集轨迹采集

`scripts/run_online_dense_collection.py` 绑定已完成a08的manifest、summary及40请求哈希链，复用上节所有前置门槛。题单、模型、主随机种子、32K主上限和21/30-token probe/补答边界均保持一致。唯一运行模式为 `dense_collect_no_stop`：每个推理候选都探测，保留 `would_stop`，但不执行置信或退化早停，继续至自然EOS或预算末端。预算末端属于明确删失，不能记为自然完整轨迹。

在新GPU服务器release中，沿用上节参数，将入口替换为 `scripts/run_online_dense_collection.py`，增加 `--development-run "$DEVELOPMENT_RUN"`（指向已完成的a08 `runs/development10-001`），并使用新的 `--run-root runs/dense10-001`。先分别执行 `--inspect-only` 和 `--preflight-tokenizer-only`，再由服务器tmux托管实际生成与独立严格判分。判分分母为 `--planned-count 10`。

总时限7200秒、每请求1800秒；没有题目、配置、seed或预算覆盖参数。每题前核对输入身份，失败即停、不自动重试，保存partial及未执行名单。保存位置为 `requests/devNN/dense-collect-no-stop/`；scope和collection-only标记排除其进入在线性能主表。生成、判分及完成回执分别核验。采集不会修订既有未决答案或人工裁决。

完成并取回核hash后，在本地运行独立离线分析：

```bash
python scripts/analyze_dense_collection.py \
  --collection-run "$COLLECTION_RUN" \
  --development-run "$DEVELOPMENT_RUN" \
  --output-dir runs/dense10-20261008/analysis-001
```

分析逐题核对Vanilla完整主采样流、原dense公共probe、首次would-stop、停止后新增观测、自然终点/预算覆盖、token守恒和计时。自适应回放只读取模拟已查询点，跨跳点累加原始分段主推理时间；密集采集中的时间不能当作稀疏在线耗时，停止后的未来观测不提供给已经停止的策略。本阶段完成后先判断可稀疏机会与成本限制，再决定是否推进30题开发筛查；不自动启动80/120选参、MATH500或AIME。

## 十题本地答案复核与机制分解（2026-10-09）

已完成结果见 [质量与机制报告](../results/development10-quality-mechanism-20261009.md)。
以下命令在 **Mac 的 CoDE 仓库根目录**运行，不需要 SSH、CUDA 或模型加载。
原结果位于被 Git 忽略的 runs，单独 clone 不包含它们。原40条包仍为人工裁决
pending；新的助手辅助结果单独保存，不能修改旧包来标记人工完成。

1. 先读 [冻结边界准则](ANSWER_BOUNDARY_REVIEW_V1.md)。两份独立上下文助手裁决
   已存于下列 `QUALITY_ROOT`，勿在查看 gold 后修改。`seal` 只读匿名案例，必须先
   成功；`prepare` 才验证并读取私有映射。下面的 replay-001 用于另存重现，若已存在
   必须选择新目录，禁止覆盖。

```bash
QUALITY_ROOT=runs/development10-analysis-20261009/quality
QUALITY_OUT="$QUALITY_ROOT/replay-001"
python3 scripts/analyze_answer_review.py seal \
  --packet runs/development10-20261008/answer-review-pending-001 \
  --policy docs/ANSWER_BOUNDARY_REVIEW_V1.md \
  --review-a "$QUALITY_ROOT/reviewer_a.jsonl" \
  --review-b "$QUALITY_ROOT/reviewer_b.jsonl" \
  --output "$QUALITY_OUT/sealed"
python3 scripts/analyze_answer_review.py prepare \
  --packet runs/development10-20261008/answer-review-pending-001 \
  --sealed "$QUALITY_OUT/sealed" --output "$QUALITY_OUT/prepared"
runs/grading-validation/venv/bin/python scripts/grade_math_answers.py \
  --input "$QUALITY_OUT/prepared/grading-input.jsonl" \
  --planned-count 40 --timeout-seconds 10 \
  --output "$QUALITY_OUT/supplementary-math-grades.json"
```

2. 固定 Q=(1,2,3,7,11,...) 比较 A（全机会/全历史）、B（仅Q/全历史 oracle）、
   C（仅Q/稀疏历史）。新分析器单独核对历史434项审计及a08/a09原始记录hash；
   不修改旧分析器或使用当前源码冒充旧GPU身份。保存的analysis-001不可覆盖。

```bash
python3 scripts/analyze_stop_opportunities.py \
  --collection-run runs/dense10-20261008/remote-a09-final/runs/dense10-001 \
  --development-run runs/development10-20261008/remote-a08-final/runs/development10-001 \
  --prior-analysis runs/dense10-20261008/analysis-001/analysis.json \
  --output-dir runs/development10-analysis-20261009/mechanism/analysis-002
```

成功条件：40条补充分母完整并保留未决；机制结果status=passed、failed_checks为空，
A/C与旧线上实际停止点或自然EOS吻合。无crossing保持null，不能把自然终点补成
阈值crossing；单独分支在OR停止后出现的crossing属于反事实诊断。D只在共同T比较，
不能累加D差推断耗时。无warm-up的时钟诊断仍待另做，本节不等于E2全部完成。
