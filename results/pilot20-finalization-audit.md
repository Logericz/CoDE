# 两题已保存补答核查

日期：2026-10-08，北京时间。范围：读取已有 q001/q002 CoDE 记录、使用原服务器本地 tokenizer 解码、独立 CPU 数学判分。没有重新生成答案，也没有覆盖原始记录或旧 grading。

## 已核实结论

| 样本 | 保存输出中的完整答案 | 独立严格判分 | 解释 |
| --- | --- | --- | --- |
| q001，algebra/566，标准答案 2008 | `\boxed{2008}` | `needs_review / reasoning_boundary_not_isolated` | 答案后又生成 `</think>`；旧的最后标记切片漏掉了已存在的正确 boxed 值 |
| q002，geometry/428，标准答案 997 | `\boxed{2992}` | `incorrect` | boxed 完整，但数值错误；后续说明截断不改变这一点 |

q001 的最后一次生成输入以固定的 `</think>…The final answer is \boxed` 结束，新增内容为：

```text
{2008}.
</think>

To find the value of $ x $ given by the expression:

$$
x = \frac{20
```

输入、输出和新增 token 数分别是 1056、1086、30；输入结束标记位于绝对 token 下标 1044，额外结束标记位于新增序列下标 6，均为零基下标。新增 30 tokens 中没有 tokenizer EOS。输出保留完整输入前缀，新增 IDs 与输出后缀逐项一致。

因此，q001 不能再描述为“30 token 补答未产生答案”。准确结论是：受控答案已完成，模型随后重复结束标记并重新解释，旧提取方式遗漏了先前答案。无需为这项核查重新运行或延长 q001。

## 判分与边界处理

本次严格判分输入保留固定受控答案前缀和全部新增 token。q001 中重复的 `</think>` 仍触发现有判分器的边界保护；没有为了取得正确分数而删除该标记，也没有回写原 `grading.json`。

“已观察到正确 boxed 值”是边界审查事实；它与“当前冻结自动判分给出 correct”是两种状态。若后续采用显式边界裁决，应另存裁决、规则版本和 token 范围，并在主测试前冻结对全部方法一致的处理规则。两题工程记录不用于估计方法准确率或在线加速。

## 来源和保存位置

从服务器取回 33 个文件，每个文件均与远端读取时生成的 SHA-256 清单一致。独立解码报告再次核对了原始 stage、environment、检查器及 tokenizer 文件身份。原始证据保存在被 Git 忽略的 `runs/remote-evidence-20261008/`，供私下迁移：

- `FETCH_MANIFEST.json` 和 `saved-evidence.tar.gz`：原始文件及逐文件 hash。
- `runs/pilot20-diagnostic-8192/`：两题 stage、配置、环境、日志及顶层记录。
- `q001-finalization-inspection.json`、`q002-finalization-inspection.json`：原 tokenizer 的完整输入、输出和新增内容解码。
- `codestop-final-answers-strict.jsonl`、`codestop-math-grades-strict.json`：新判分输入与结果，不覆盖旧报告。
- `current-server-inventory.json`：本次只读环境清单。

关键文件 SHA-256：

```text
f79623b9848c70da6f2c42b1cdba4228911e4876d31cb2d3ef6f2c6ba09662bc  q001/codestop.json
a8e28f1c3a448c27079f342c36a31ae6e4da76a862bd48f6f0790472b41fed5b  q002/codestop.json
1803968633e17fccb41596f69fb17646d55d36ed7d966a4c074f51fa294ec8cd  saved-evidence.tar.gz
```

## 当前服务器与下一步

已有 SSH 控制连接有效并已成功复用。现场核验为 RTX 4090、24564 MiB 显存，Python 3.12.3、torch 2.9.1+cu128、transformers 4.51.3；CUDA 和 BF16 支持检查通过。固定 Qwen3-4B revision 的三个权重分片及已冻结数据仍在。检查时 GPU 计算进程列表为空。

这不是共同在线后端的 GPU 验收。下一步是补齐有界运行入口，接入答案区间和判分，再用单题检查真实 BF16 probe 数值、KV／随机流隔离和完整耗时；通过后才扩大到固定开发题。32K 容量及当前 4090 的实际速度仍待测。
