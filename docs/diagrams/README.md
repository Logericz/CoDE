# 代码阅读流程图索引

每读完一个函数或代码块，在对应 Notion 笔记旁补充流程图。先核对实现，再绘制输入、处理、分支、返回和保存边界；已有图仍准确时更新或引用原图。图的完成不自动代表学习验收通过。

## 已补齐的内容

2026-10-06 按历史阅读记录补齐以下七张图，原答案和任务状态保持不变。

| 图 | 对照代码 | Notion 对应笔记 |
|---|---|---|
| [01 普通逐步推理](code-reading/step-by-step.png) | `method_prompts.py::method_step_by_step()` | [9/22 D1：第1节](https://app.notion.com/p/3e3cbe2923d68163867cee75efd8eb17) |
| [02 消息到模型输出](code-reading/generate.png) | `models.py::generate()` | [9/22 D1：第2节](https://app.notion.com/p/3e3cbe2923d68163867cee75efd8eb17) |
| [03 方法注册、依赖、判分与保存](code-reading/inference-entry.png) | `inference.py` 注册表、依赖表、`main()` | [9/23 D1：学习笔记入口](https://app.notion.com/p/3e4cbe2923d681209076fe8160488d50) |
| [04 强制补答](code-reading/force-answer.png) | `inference.py::generic_forceans()` | [9/29 D3：补答笔记](https://app.notion.com/p/3eacbe2923d681439208ca96aab0e077) |
| [05 教学答案抽取与判分](code-reading/grade-response.png) | `src/diagnostic_core.py::grade_response()` | [9/29 D3：教学判分与正式评测的区别](https://app.notion.com/p/3eacbe2923d681439208ca96aab0e077) |
| [06 run、grade、inspect 与保存](code-reading/run-grade-inspect.png) | `main()`、`execute_stage()`、`grade_run()`、`inspect_run()` | [9/29 D3：我的原答案第3项](https://app.notion.com/p/3eacbe2923d681439208ca96aab0e077) |
| [07 重试判分与哈希复核](code-reading/regrade-hash.png) | 已学习的 `grade` / `sha256sum` 操作块 | [9/29 D3：证据位置与哈希](https://app.notion.com/p/3eacbe2923d681439208ca96aab0e077) |

已存在的 [DEER 总流程与缓存分支两张图](method-deer/README.md) 保留在 [D4](https://app.notion.com/p/3e2cbe2923d68173800cfb6e41362822)。这些图可辅助继续阅读；尚未据此把 DEER 全部阅读或实验标为完成。复习页使用链接回到原笔记，避免多份图片漂移。

## 图的边界与源码细节

2026-10-07新增：[置信度辅助函数的完整循环与算例两张图](confidence-probe/README.md)，对应D4的 `calcu_max_probs_w_kv()`。图说明首尾概率、分母与短试答边界，保留阅读与真实运行验收的区别。

- 前四图依据当前本地上游带注释工作树，后三图依据当前教学封装和已学命令。文件指纹见 [source-audit.json](code-reading/source-audit.json)。上游注释未在本次修改或提交。
- 官方主入口是生成、判分、再写 JSONL；教学封装先保存生成记录，再判分。图03/06明确分开，勿混用。图03概括执行职责，省略分块与文件名参数等准备细节；图06省略错误诊断写入路径。
- `generic_forceans()` 用长度条件决定是否补答，不是先判断答案正确性。`tokens_left`、`remainder_tokens` 在添加结束标记和答案前缀之前计算；不能把 `response_tokens` 视作最终文本完整重编码长度，也不能由此保证最终回答严格不超过 B。删除到列表空时仍沿用最后一次计算值，图中保留这一实现边界。
- `grade_response()` 的 `correct=None` 是待复核，`checked` 下的 `False` 才表示判错；不是正式数学基准评估器。
- SHA-256 复核支持生成文件在两次检查间内容不变，不证明答案正确或完成异地备份。图内数字和拼接例子用于教学，不是新增实验结果。

## 后续每次阅读完成后的处理

1. 根据用户实际汇报确认函数名或代码范围，不把准备阅读当作完成。
2. 核对当前源码与调用者，选择流程图、变量拼接图或分支图；补充易混点。
3. 保存可编辑图源及 PNG/SVG，检查中文、箭头、条件和返回字段。
4. 读取当前 Notion 页，在对应笔记旁插图，保留原答、纠正与完成状态，再读回确认。
5. 更新本索引并同步 GitHub。若没有对应学习页，先查找实际学习记录，不凭旧日期写错位置。

## 重绘

`code-reading/render.py` 集中定义七张图的节点、连线与颜色；同目录 `.dot` 是可编辑 Graphviz 文件，`.svg` 适合放大，`.png` 用于 Notion 图片块。

在项目根目录执行：

```bash
python3 docs/diagrams/code-reading/render.py
```

作用：从已核对的节点和连线生成七组 DOT、SVG、PNG。需要 Graphviz 的 `dot` 和中文字体（当前为 macOS 的 PingFang SC）。此命令只重绘文档，不调用模型、不自动上传 Notion。
