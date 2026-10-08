# 独立数学判分协议

本模块实施 `manuscript/coling2027/EXPERIMENT_PLAN.md` 第 10 节的本地数学判分，不修改原始推理记录，也不代表已完成 GPU 实验。它独立于 `diagnostic_core.py`、pilot 和 upstream adapter，避免改变已冻结运行的身份。协议版本为 `math-answer-only-v1`。

## 输入与答案边界

输入为 UTF-8 JSONL，每个**已执行请求**一行，`id` 必须是唯一非空字符串。必需字段：

- `gold`：干净的 MATH LaTeX 答案，不附题目、解题过程、`$` 数学分隔符或 boxed 包装。
- `answer_text`：调用方明确隔离的最终答案区间；允许空字符串以保留失败记录。判分器不从整段模型回复推断此区间。
- `id`：请求唯一身份；题目、方法、rollout 应由调用方组成不同请求 ID。

可选字段：`execution_status` 为 `completed`（默认）或 `failed`；`stop_reason` 原样记录；`source_sha256` 为原始证据文件的 SHA-256；`boundary_policy` 描述边界来源，例如已保存的答案前缀 token 范围与新增答案 token 范围。省略来源时报告 `provenance_status=unknown`、省略边界说明时报告 `boundary_policy=unknown`。即使提供了 hash，判分器也只记录调用方声明，不宣称独立认证原始证据或区间来源。

**注入前缀属于答案区间的一部分。** 若 runner 已注入 `The final answer is \boxed`，新生成 token 只有 `{2008}`，调用方必须提供已保存的答案前缀与生成内容构成的 `\boxed{2008}` 区间，不能仅提交新 token，也不能在结果出来后根据 gold 补造前缀。边界证据应能指向原始 token／文本记录。不要用 `rsplit('</think>')` 从任意整段回复中选取一段；重复 thinking 标记或异常关闭必须先做边界审计。

出现 `<think>`、`</think>`、analysis／reasoning 标签及相应渠道标记（包括部分未闭合和重复标记）的文本直接记为 `needs_review`，不会扫描其中的 boxed 或正确数字。调用方漏掉所有标记却仍把思考全文放进 `answer_text`，无法由本模块自动证明其区间错误，因此边界审计仍是上游责任。

## 抽取与比较

1. 在已隔离区间内，以平衡花括号识别**最后一个完整顶层** `\boxed{...}`，处理分数等嵌套花括号与集合的转义花括号。先前 boxed 即使正确也不回退选用。boxed 内部另有 boxed 时不把内层提升为最终答案；选中的顶层 boxed 含嵌套 boxed 则记 `needs_review`，当前协议不对其做隐式化简。
2. 若最后完整 boxed 后还有未闭合 boxed，仍按现协议选择最后完整 boxed，并保存未闭合位置、`has_unclosed_tail` 与汇总数量。这是预定抽取规则，不等同于断言尾部无歧义。
3. 无完整 boxed 且出现 boxed 残片：`malformed_box`。空答案或选中的空 boxed：`empty_answer`。没有 boxed 时只接受单个裸数学表达式或一个完整数学分隔区间；无 boxed 的说明性自然语言送人工复核，不从中搜索一个有利数字。
4. 用 **`math-verify[antlr4_13_2]==0.9.0`**，强制 ANTLR runtime 4.13.2、latex2sympy2_extended 1.11.0、SymPy 1.14.0、mpmath 1.3.0；不符合版本时中止 CLI。所有已安装包版本另写进 manifest。只使用 LaTeX 提取，`fallback_mode='no_fallback'`、`extraction_mode='first_match'`，禁用 boxed 二次提取。选中表达式包装为单一 `$...$` 后解析，不允许内部 `$` 注入额外提取候选。字符串 fallback 与多提取结果不作为已验证答案。
5. 方向固定为 `verify(gold, prediction)`。`strict=True`、`float_rounding=6`、`numeric_precision=15`、`allow_set_relation_comp=False`。这保留 Math-Verify 的数值容差和方向性，不是严格形式化证明。LaTeX normalization 明确启用 basic_latex，关闭 units、malformed_operators、nits、boxed、equations 变换。
6. 每题新启一个 CPU 子进程。`parse(parsing_timeout=None, raise_on_error=True)` 与 `verify(timeout_seconds=None, raise_on_error=True)` 关闭库内部定时器，由外层硬超时杀死子进程。默认 10 秒，包含解释器启动、导包、解析与比较；外层超时返回 `timeout`，不会变成 `incorrect`。选项与源文件 hash 进入 manifest。

Math-Verify 0.9.0 的 `utils.timeout(None)` 返回无计时包装器；`parse` 与 `verify` 默认会捕获部分异常／超时并返回空列表／False，因此本模块显式禁用其 timer、打开 `raise_on_error`。库内部某些符号运算仍有自己的异常回退，报告的 correct／incorrect 含义是**冻结版本和参数下的判分结果**；需人工核验的案例另记，不改原始输出。

## 状态和分母

`correct` 与 `incorrect` 表示成功解析后的等价比较结果。`empty_answer`、`malformed_box`、`needs_review`、`gold_parse_failure`、`prediction_parse_failure`、`unsupported_type`、`timeout`、`evaluator_error`、`dependency_error` 均保存 `grade=null`，列入待复核；不会伪装为数学错误。调用方报告执行失败时记 `run_failure`，保留在已执行分母，不尝试从残留文本补救答案。预算用满本身不判错，有有效最终答案则照常比较。

CLI 必须给出 `--planned-count`。输入行数是已执行数量；计划数不能小于输入行数，未执行请求不得伪装成失败输出，也不得因未执行而把矩阵称为完整。主指标为 `confirmed_correct_count / executed_count`，附 `unresolved_count` 与 `(confirmed_correct_count + unresolved_count) / executed_count` 的宽松上界。执行失败保留分母，但没有答案的运行失败不加入“待复核答案全对”的分子。零已执行时准确率为 null。状态计数、执行失败、未执行、`unknown_boundary_policy_count` 和 `unclosed_tail_count` 分别报告。重复 ID、重复 JSON 键、NaN／Infinity、空行与非法 JSON 会中止，不静默覆盖或丢弃。

待复核记录应在另一个去除方法身份的人工复核工件中处理；本 CLI 不回写裁决，也不将复核后的数字自动混入旧报告。

## 使用与验证

在项目目录中执行；仅 CPU 判分，无 GPU 或网络调用：

```bash
runs/grading-validation/venv/bin/python scripts/grade_math_answers.py \
  --input runs/my-run/final_answers.jsonl \
  --planned-count 20 \
  --timeout-seconds 10 \
  --output runs/my-run/math_grades_v1.json
```

省略 `--output` 则 JSON 发往 stdout。指定输出必须是不存在的新文件；拒绝输入原路径、已有文件、符号链接，不修改输入。报告保存输入文件 hash、Python／依赖版本、判分模块与 CLI／库 parser、grader、utils 源文件 hash、抽取与比较参数。输入在读入和判分过程中改变会中止，避免把改变后的 hash 配给旧结果。

```bash
runs/grading-validation/venv/bin/python -m unittest discover \
  -s tests -p test_math_grading.py -v
```

测试包含真实 Math-Verify 的整数、分数、根式、集合、区间、错误答案、解析失败，以及思考区间拒绝、最后 boxed、未闭合尾部、完整分母、重复 ID、子进程超时、只写新报告。未安装指定包时真实集成测试明确跳过，其他测试仍可执行；跳过不是集成通过。
