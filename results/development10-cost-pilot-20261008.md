# 固定10题开发成本试跑

日期：2026-10-08，北京时间。接续提交 `8b727ac`。本轮属于开发成本与执行诊断，不做阈值/日程选参，不进入论文主测试。

## 计划与当前状态

**16:11:18已在GPU服务器tmux启动，当前运行中，完成数量与判分以远端最终工件为准。** 固定10题×4配置，共40个计划请求；没有承诺两小时内全部完成。总墙钟上限7200秒，单请求1800秒，两者同时生效，操作之间检查。失败、OOM或超时停止矩阵、保存partial，其余请求记未执行；不自动重试。

题目来自冻结pilot20（属于selection120开发划分）。按原ID排除历史已暴露的 `math/train/algebra/566` 与 `math/train/geometry/428`，再取冻结原顺序前10题。历史q001/q002位于JSONL第3/9行，并非前两行。题单在运行前保存，未按答案、题目长度或耗时选择；本轮暴露归入开发记录。

| 新编号 | 来源行（0-based） | 原题号 | prompt tokens |
| --- | ---: | --- | ---: |
| dev01 | 0 | `math/train/algebra/211` | 47 |
| dev02 | 1 | `math/train/algebra/526` | 65 |
| dev03 | 3 | `math/train/algebra/581` | 59 |
| dev04 | 4 | `math/train/algebra/940` | 378 |
| dev05 | 5 | `math/train/counting_and_probability/644` | 79 |
| dev06 | 6 | `math/train/counting_and_probability/720` | 134 |
| dev07 | 7 | `math/train/geometry/276` | 70 |
| dev08 | 9 | `math/train/intermediate_algebra/238` | 88 |
| dev09 | 10 | `math/train/intermediate_algebra/453` | 72 |
| dev10 | 11 | `math/train/intermediate_algebra/712` | 90 |

每题固定Vanilla、dense CoDE、fixed(interval4)、adaptive既有默认配置，按题号循环换序，10题下顺序位置近似均衡。同题共用master42/rollout0派生的主随机种子，每请求重置KV、控制器及私有采样流。主上限32768、probe最多21、最终补答最多30，BF16/eager，模型与tokenizer revision沿用固定版本。完整参数和config hash在运行manifest内。

最大prompt378tokens；连同32768与42预留需要33188，小于context40960。dev04比旧32961合成容量多227个位置；旧容量不证明这部分实际显存，本轮按真实运行观察，失败即停，不筛掉该题。

## 实现、测试与部署

- 新增 `scripts/run_online_development.py`，复用既有same-KV/自然边界/合成容量门槛，另绑定已完成a07原始记录。131项输入与源码身份通过远端纯校验，10题/40请求与运行前题单完全匹配。
- 新增11项多题关键路径CPU测试全部通过；本机固定环境全套273项，269通过、4因缺判分依赖跳过。严格判分代码未修改，服务器固定Math-Verify环境将另行判分。
- 远端独立a08部署20个文件，上传前后SHA逐项一致；含新入口、固定参考源码及私有tmux托管文件。旧已运行的release不覆盖。
- tokenizer-only预检未导入Torch或加载权重，10题全部满足实际context限制。真实运行再次核查全部prompt。

## 服务器托管与恢复

用户要求Mac可能休眠，因此任务由**服务器tmux**托管；服务器已安装tmux3.2a，仅新增该系统包，未升级Python/模型依赖。独立会话 `codestop-dev10-20261008-a08` 于16:11:18创建；另一次SSH读取确认后台Python及GPU进程存在，16:11:52首请求dev01/Vanilla已到640tokens。模型加载和预热完成，未与CPU测试混淆。

私有supervisor顺序执行生成、独立严格判分、完成回执；即使生成非零退出，只要已发布最终答案输入，也按planned-count40判分所有已执行行。各步骤独立退出码，不把生成完成当作判分完成。异常时清理并等待自己启动的进程组，最后核对部署身份；旧结果目录和已有回执拒绝再次使用。

重新连入原服务器后：

```bash
tmux attach -t codestop-dev10-20261008-a08
# Ctrl-b 然后 d：离开查看，继续运行。
```

远端a08根目录保留 `console.log`、`grading-console.log`、`supervisor-start.json`、`generation-exit.json`、`completion.json`。生成结果在 `runs/development10-001/`，含events、每请求原子文件、增量answers日志、最终答案JSONL和summary。Mac休眠/SSH断开不会结束tmux；服务器自身停机不在此保障范围。连接凭据及机器专有命令只保存在忽略目录，不进入Git。

## 判分与分析边界

原严格 `math-answer-only-v1` 不变；重复think仍needs_review，不根据gold修补答案。原人工复核包6条全pending，这不等于严格判分有6条unresolved。成本试跑可以继续，但不据质量未决的更快请求宣称准确率保持或选择参数。

结束后分别报告计划/执行/失败/未执行、各配置覆盖、完整四配置配对、主/探测/补答tokens、完整耗时和峰值显存，以及严格C/E、(C+U)/E与未决数。超时留下的完整题存在幸存选择，不能代表全部10题或MATH500；固定前10题也不保证覆盖所有题型/长短轨迹。

## 证据保留

启动前保存本地163个文件基线、远端496个既有源码/数据/结果基线。原始计划、环境、部署清单、实时日志和私有托管代码位于忽略目录 `runs/development10-20261008/`。旧学习注释、稿件、冻结数据和原评分不改写。本轮完成后另核对保留状态。
