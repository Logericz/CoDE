# DEER 试答与置信度：源码阅读图

核对日期：2026-10-07。来源：本地带注释的 `upstream/CoDE-Stop/method_deer.py::calcu_max_probs_w_kv()` 与教学封装 `src/upstream_diagnostic.py::Backend._guard_probe()`。文件指纹见 `source-audit.json`。

图片已供 [Notion D4](https://app.notion.com/p/3e2cbe2923d68173800cfb6e41362822) 阅读使用。图和下面的算例属于助手整理的材料，不是学生独立作答，不代表真实模型运行或阶段验收。

- [完整试答循环](overview.png)：输入、临时KV、逐token选取、记录、停止、返回。可放大的 [SVG](overview.svg) 和可编辑 [DOT](overview.dot) 同目录保存。
- [概率保存与评分算例](score-example.png)：三token示例与极短试答边界。[SVG](score-example.svg) / [DOT](score-example.dot)。

## 源码对应

当前注释使行号移动，以函数名和语句定位为准：12–42行准备停止token与缓存；44–62行前向预测；64–80行累计、保存、更新和步数判断；82–89行结束标志与分数；91–111行解码、清理、返回。

图中 `n` 对应 `total_steps`，`S` 对应循环内尚未做最终变换的 `total_prob_max`，`p` 对应当轮 `max_value`。

1. `method=0` 使用编码后的 `</think>` ID；`method=1` 增加换行与标点对应ID。源码判断单个ID属于列表，不做多token完整字符串匹配。`ended_with_think` 在method=1时不一定专指 `</think>`。
2. `cp_cache=False` 的调用输入是题目、保留推理、试答前缀三段；helper内部仍更新并使用本次试答的KV。传入已有KV时先深拷贝，临时答案的KV不应混入保留的推理缓存。
3. 试答选择概率最大的token（贪心），不同于上层正式答案生成的采样调用。每轮都保存ID和概率；首轮不向分数累计值加入概率。
4. 步数判断是 `n > tot_step_cap`，cap为20时可生成21个token。循环后总是减去最后一项，达到上限时也减，因此最后一项不保证是停止token。
5. helper返回字典；阈值比较在 `method_deer()` 中，写文件由外层程序处理。图省略无返回用途的 `prompt_text` 调试变量和CUDA分配器缓存清理，只保留临时KV生命周期。

## 公式逐项对照

设生成n个token，概率为p1…pn，以下推导假设n≥2且概率为正。

- method=0：循环内 `S = log(p2)+...+log(pn)`；返回 `exp((S-log(pn))/(n-1))`，即 `(p2×...×p[n-1])**(1/(n-1))`。
- method=1：循环内 `S = p2+...+pn`；返回 `(S-pn)/(n-1)`，即 `(p2+...+p[n-1])/(n-1)`。
- 两种情况下，分子保留n−2项，分母却是n−1；不能直接称为保留项的普通几何/算术平均。这里描述现有实现，没有修改论文方法或源码。

教学例：method=0，概率 `[0.70, 0.81, 0.64]`，第3个token为停止token。概率列表保存全部三项；累计时省略第1项、结束后减去第3项，最终 `exp(log(0.81)/2)=0.90`。这是公式代入，不能解释为90%的答对概率。

仅1个token时分母为0；仅2个token且概率为正时，method=0返回1、method=1返回0。教学封装在返回结果后先保存诊断事件，再拒绝≤2 token或非法分数等情况，不将其静默修正成可用置信度。

已有 `tests/test_upstream_diagnostic.py::OriginalProbeTests` 的5项CPU合成测试在本轮先前核验通过。覆盖普通三token、1/2 token、21步、EOS和非有限分数；这不覆盖真实GPU兼容性或模型效果，本次绘图未重跑模型。

## 重绘

在项目根目录执行：

```bash
python3 docs/diagrams/confidence-probe/render.py
```

此命令复用 `code-reading/render.py` 的Graphviz样式，生成本目录两组DOT/SVG/PNG，不加载模型、不自动上传Notion。需要Graphviz和中文字体，当前使用PingFang SC。
