# 共同在线入口：单题 GPU 验收

日期：2026-10-08，北京时间。结论：有界入口已落地，q002 的三个在线请求已完成；单一保存前缀的同KV原函数对照、历史在线结果重现及0/1/2 probe隔离通过。完整前缀 probe 的严格数值对齐失败，**暂不扩大实验**。此报告不是论文准确率或速度主表。

## 环境、范围与保留

运行前经 SSH 重新核验 RTX 4090、24564 MiB、驱动 580.105.08、Python 3.12.3、torch 2.9.1+cu128、CUDA 12.8、transformers 4.51.3，CUDA/BF16 可用，GPU 无其他计算进程。沿用原 `.venv`、固定 Qwen3-4B/tokenizer revision `1cfa9a7208912126459214e8b04321603b3df60c` 和已有冻结数据，没有重装依赖或下载模型。

部署到独立目录 `codestop-online-acceptance-20261008-a01`；原 prep 项目、旧教学/pilot 结果与学习注释不覆盖。a01 的 11 个部署文件逐项核对 SHA-256，部署清单 hash 为 `411d8111e7f72122e3ab698d669e0c20fb1072297ed4342b9999ec7d473d8aa9`。

冻结数据 manifest hash 为 `ae236d14cfa3b5cfbf434163386fea878016355051313ccdbd11d0a309eebca7`，sources lock hash 为 `ded0add6ea18a0f36e7433fe4d434b29577795bd7ee4acdee70a3e1f2a08415e`。本次没有重抽样。q001 只读复用保存前缀检查 probe，不生成完整解答寻找 2008；此前的正确 boxed 审查与旧严格 needs_review 均保留。

## 实现变化

- 新入口 `scripts/run_online_diagnostic.py`：固定一题/rollout、显式 token 上限、冻结数据/代码身份、物理 GPU UUID 锁、预热单列、64-token 实时进度、逐请求原子保存、失败部分记录和完整执行分母。
- 新验收 `scripts/validate_online_gpu.py`：从固定上游 Git object/同 hash 导出源码提取参考函数；按保存 prompt 和逐 token teacher forcing 构建真实在线 KV。预先固定 `atol=rtol=0`，首次差异落盘并退出。
- 修复第二 EOS 漏识别：固定模型的 generation config 声明 `[151645,151643]`。主推理、最终补答和 probe 异常判断均采用完整集合；协议为 `common-online-validation-v2`，引擎 `1.1.0`。
- 判分从保存的答案 token 区间解码，仅移除一个末尾已知 EOS，其余符号保留；未确认答案边界送 `needs_review`。重复 think 不删，旧判分不回写。

## 严格数值验收：失败

13:54 开始，三份保存前缀的来源与 tokenizer 校验通过。首个 q001:0 在 prompt＋已保存推理 257 tokens 后做 probe，与相同当前 eager/BF16 模型上的固定上游完整 268-token 输入比较。

| 检查 | 结果 |
| --- | --- |
| 21 个贪心 probe token IDs | 完全一致 |
| confidence | 两者均 0.921875 |
| ended_with_think | 两者均 false |
| probe 前后主 KV、logits、私有/全局 RNG | 指纹不变 |
| 原始 BF16 token 概率 | 4 项不同，exact 失败 |

概率下标为零基：

| 下标 | 在线 KV＋新 probe | 原函数完整前缀 | 在线减参考 |
| ---: | ---: | ---: | ---: |
| 12 | 0.9375 | 0.91796875 | 0.01953125 |
| 14 | 0.99609375 | 1.0 | -0.00390625 |
| 15 | 0.62109375 | 0.6796875 | -0.05859375 |
| 18 | 0.9765625 | 0.96875 | 0.0078125 |

这是当前同一个 eager/BF16 模型下两条计算路径的比较，不能归因于历史保存记录使用 SDPA。最大差并非一个 BF16 末位。该运行未执行后两个前缀的数值比较，也未到达后续 0/1/2 probe 续写检查。原失败保留；后续同 KV 定位不能将它改为通过。

## 同KV定位与隔离：局部通过，原数值门槛仍失败

14:02–14:03，在独立a02部署运行 `scripts/diagnose_probe_parity.py`，模型/精度/attention/源码/前缀/seed均与a01核对，原完整前缀结果C直接复用并验证hash。只新增以下A、D、B三条比较probe，均21 tokens；不放宽容差。

| 路径 | 计算方式 | confidence |
| --- | --- | ---: |
| A | prompt预填充＋逐token接纳保存的推理；新probe | 0.921875 |
| D | 与A同一份主KV；固定原函数，含原deepcopy/cache_to_device | 0.921875 |
| B | 257-token前缀整段预填充；新probe | 0.92578125 |
| C | a01原函数完整268-token输入，复用已保存结果 | 0.921875 |

- **A=D exact通过**：全部token、原始概率、confidence、ended一致，双方操作前后同一主KV/logits/RNG指纹未变。本次A也与a01在线结果完全一致。
- **A与C仍失败**：重复观察到原4项概率差异。
- **B与C失败**：5项概率不同，最大绝对差0.0625，confidence也不同。A与B比较亦失败；四条路径的probe token IDs仍全部一致。
- A与B的36层KV以及末尾logits指纹不同。这把差异定位到前缀构建/分段计算路径，但没有证明具体kernel或BF16舍入机理，亦不证明跨样本停止决策不受影响。

独立隔离使用同一保存前缀，插入0/1/2次probe后各续写64 tokens。三个variant全部通过：主采样token和概率、最终KV/logits、私有与全局RNG精确一致；每次插入probe前后主状态也未变。它验证此病例的KV/RNG隔离，不能替代后两个保存前缀或全部运行分支。诊断正常完成，按设计退出码2保留严格失败状态。

a02包含验收工具的一处EOS接线完善：隔离续写采到EOS时记录后直接结束，不将它再写入KV，与engine一致。a01在到达隔离阶段前已失败，这项后续改动不改变其原数值证据。本次三个隔离variant均实际续写满64 tokens；并未借此验收自然EOS。

## q002 在线入口烟测：3/3 完成

13:55–13:58，固定 `math/train/geometry/428`，master seed=42、rollout=0，派生主种子 `1937708343865469216`。每配置主上限 1024、贪心补答上限 30，BF16/eager。模型加载 5.6829 秒，独立预热请求 2.4797 秒，均不计入下面请求时间。预热未覆盖 probe。

| 配置 | 主 tokens | probe 次数/tokens | 补答 tokens | 实际生成总数 | 请求秒 | 峰值分配显存 bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Vanilla | 1024 | 0/0 | 30 | 1054 | 41.0388 | 8320695808 |
| CoDE dense | 1024 | 1/21 | 30 | 1075 | 41.6355 | 8320695808 |
| CoDE fixed，间隔4 | 1024 | 1/21 | 30 | 1075 | 42.2210 | 8320695808 |

三配置均遇到一个候选：第300个采样 token 为 pending Wait，probe 使用此前299个主 tokens的前缀。dense/fixed 的 probe confidence=0.81640625、ended=false，继续主流，最终 `stop_reason=budget`。两者尚处共同密集预热阶段，未发生任何稀疏跳过。

三个请求的主采样记录（含 token 和采样/原始概率）及最终输出 token 完全一致；插入一次 probe 未改变本条主轨迹。token 守恒通过，分项时长与总时长闭合，代码哈希运行前后相同。受控 final prefix 的输出区间 `[1024,1036)`，答案区间 `[1025,1066)`（零基、半开），包含 boxed 前缀；没有 EOS 被剥除。

独立 Math-Verify 0.9.0 判分在原固定依赖 CPU 环境完成，计划/执行=3/3、运行失败=0、待复核=0。三个配置都输出 `\boxed{3992}`，标准答案997，均 `incorrect`。这是同一开发题的三个短预算请求，不是三个独立测试样本，不汇报泛化准确率。

覆盖：在线 prefill、采样、Wait→probe→继续、KV 分支、思考预算耗尽、受控补答、答案导出、完整计时、独立判分。未覆盖：实际在线早停、自然 EOS、自然进入答案后预算耗尽、32K 容量、主表精度或调度收益。

## 原始证据

原始文件保存在忽略的 `runs/online-handoff-20261008/`，不直接提交 Git：

- `remote-preflight.json`、`local-preservation-before.json`、`remote-preservation-before.json`。
- `release-a01/`：冻结执行源和部署清单。
- `remote-a01-final/FETCH_MANIFEST.json`：26 个文件与远端 SHA-256 全部一致。
- `remote-a01-final/runs/gpu-validation-001/`：原始 probe 概率/BF16 bits、状态指纹、首差、错误及事件日志。
- `remote-a01-final/runs/online-q002-1024-001/`：manifest、环境、预热、三个请求、答案导出、事件日志与独立判分。
- `release-a02/` 与 `remote-a02-final/runs/parity-diagnosis-001/`：定位源码、A/B/D和复用C、状态指纹、各比较和三组隔离记录。a02的26个证据文件与远端SHA-256全一致；部署13文件也全部核验。

a01 取回压缩包 SHA-256：`be6b8419688283acc6ebc21455b10be88cf0ac9d33fa4bd16da00b41b210d605`。在线最终答案 JSONL SHA-256：`5880a9cd0fcdb8a8ab3d4bb6f495441fafeb26bba16719941d2e4dab6d0b22a8`。

a02部署清单SHA-256：`1381a323ea35a4e28778b872c826b0684839591b552203b5db50db02554d2fee`；取回压缩包SHA-256：`61d73befc99adf0b082d58211b59a8db171d2c26dec7f5b96e7634be08c4a36d`。最终复核185个远端原文件及45个本地保留文件，内容hash均与运行前相同。运行结束后GPU计算进程列表为空。

## 本地检查与接续决定

- 当前本地完整发现范围运行175项：171通过，4项因该环境未装判分依赖而明确跳过。此范围包含原有未跟踪的pilot/数据准备测试，不能视作仅克隆Git即可得到的范围。
- 专用固定判分环境运行25项全部通过，包含上述真实Math-Verify集成。远端三个在线答案也完成真实独立判分，无跳过。
- 最终测试日志为 `local-unittest-final-a02.log`；源码、入口、验收、定位及判分测试通过不等于完整GPU验收通过。

当前决定：保持单题工程范围。下一步明确正式在线probe应以同KV原函数还是完整前缀回放为数值参考，评估跨路径差异对阈值/停止判定的影响，并补齐剩余GPU分支；冻结契约后再进入正式32K容量及小规模测速。不要把同KV局部通过改写成旧完整前缀exact通过，不依据本次短预算时长外推完整实验费用。论文主表继续TBD，稿件未修改。
