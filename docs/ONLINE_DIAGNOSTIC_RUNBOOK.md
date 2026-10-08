# 有界共同在线入口与 GPU 验收

范围：一题、一个 rollout、顺序运行 Vanilla / dense CoDE / fixed CoDE。默认使用已暴露的 pilot20 q002 (`math/train/geometry/428`)；首次主预算 1024 tokens，每次最终补答最多 30 tokens，每个 probe 最多 21 tokens。此入口最多允许 8192 主 tokens，不是论文完整矩阵或 32K 容量验收入口。

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
