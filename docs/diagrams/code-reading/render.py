"""Render the source-audited learning diagrams; requires Graphviz and PingFang SC."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
GRAPHS = {
    "step-by-step": (
        "01 · method_step_by_step()\n把一道题包装成消息，再取回回答",
        [
            ("a", "输入：model、tokenizer、sample、B\n只从 sample 取 question，参考答案不送入模型", "input"),
            ("b", "content（字符串）\n题目 + 逐步推理要求 + 最终答案格式要求", "normal"),
            ("c", "messages（列表）\n[{role: user, content: content}]", "normal"),
            ("d", "模型名或路径包含 nemotron？", "decision"),
            ("e", "在消息列表前加 system 消息\ndetailed thinking on", "normal"),
            ("f", "调用 models.py 的 generate()\n传入 messages、B；do_sample=True", "call"),
            ("g", "取得 response（回答字符串）\n重新 encode → response_ids → len", "normal"),
            ("h", "返回字典\nresponse / messages / response_tokens / work\n这里 work = response_tokens", "output"),
            ("i", "边界：只返回，不判分、不保存\n重新编码的 token 数不等同于实测耗时", "note"),
        ],
        [("a","b",""),("b","c",""),("c","d",""),("d","e","是"),("d","f","否"),("e","f",""),("f","g",""),("g","h",""),("h","i","")],
    ),
    "generate": (
        "02 · models.generate()\n消息如何变成张量，再变成回答",
        [
            ("a", "输入：messages（列表）、B、模型与分词器\nreturn_full_text 默认 False", "input"),
            ("b", "apply_chat_template(messages)\ntokenize=False；add_generation_prompt=True\n得到 prompt（字符串）", "normal"),
            ("c", "tokenizer(prompt, return_tensors=pt)\n得到 inputs：input_ids、attention_mask 等张量\n移到模型参数所在设备", "normal"),
            ("d", "构建 gen_kwargs\nmax_new_tokens=B + pad/eos ID + kwargs\n后面的 kwargs 可覆盖同名设置", "normal"),
            ("e", "torch.no_grad() 下调用 model.generate()\noutputs：输入 token + 新生成 token", "call"),
            ("f", "return_full_text 为 True？", "decision"),
            ("g", "取 outputs[0]\n保留输入前缀与新增部分", "normal"),
            ("h", "input_length = inputs[input_ids].shape[1]\n取 outputs[0][input_length:]\n只保留新增部分", "normal"),
            ("i", "tokenizer.decode(..., skip_special_tokens=True)\n返回 response（字符串）", "output"),
            ("j", "B 是新增 token 上限，不是必须生成的长度\n教学例：输入120、总输出170 → 默认解码后50个位置", "note"),
        ],
        [("a","b",""),("b","c",""),("c","d",""),("d","e",""),("e","f",""),("f","g","是"),("f","h","否 / 默认"),("g","i",""),("h","i",""),("i","j","")],
    ),
    "inference-entry": (
        "03 · 官方 inference.py 主入口\n查表选方法 → 检查依赖 → 生成 → 判分 → 保存",
        [
            ("a", "命令行参数 args\nB = args.max_new_tokens", "input"),
            ("b", "method_fn = METHOD_REGISTRY[args.method]\nstep-by-step → method_step_by_step 函数对象\n此时只是选函数，尚未生成", "call"),
            ("c", "按 dataset 选择 validate_fn\n若用 LLM 判分：partial 预先绑定客户端等参数\n加载数据、模型；读取已处理项与依赖缓存", "normal"),
            ("d", "方法有依赖，但整份缓存为空？", "decision"),
            ("e", "抛出 RuntimeError\n提示先运行前置方法", "error"),
            ("f", "遍历 (sample_idx, rollout_idx)\n已处理项跳过；进入下一项", "normal"),
            ("g", "本项需要依赖，且缓存缺这一项？", "decision"),
            ("h", "打印警告并跳过这一项\n继续处理下一项", "note"),
            ("i", "调用 method_fn(...) → method_output\n有依赖时额外传入 sample_response", "call"),
            ("j", "validate_fn(response, sample[answer])\n得到 validation", "call"),
            ("k", "组装 result 字典\n方法输出 + 判分 + 参数 + 样本与编号", "normal"),
            ("l", "f.write(json.dumps(result) + 换行)\nf.flush() → 继续下一项\n输出文件：每行一个结果的 JSONL", "output"),
            ("m", "基础依赖：step-by-step → step-by-step-forceans\n再分为 deer / codestop 两支；两者不是前后依赖\nflush 刷出 Python 缓冲，不等同于 fsync 持久化保证", "note"),
        ],
        [("a","b",""),("b","c",""),("c","d",""),("d","e","是"),("d","f","否"),("f","g","未处理项"),("g","h","是"),("g","i","否"),("i","j",""),("j","k",""),("k","l",""),("l","m","")],
    ),
    "force-answer": (
        "04 · generic_forceans()\n保留一部分推理，给模型一个答案开头让它续写",
        [
            ("a", "读取前置记录中的 response 与 messages\n原回答 encode 后计数；预留量 M：数学50 / 代码300", "input"),
            ("b", "原回答 token 数 < B − M？", "decision"),
            ("c", "直接返回原回答；forced=False\nremainder、final_part 为空\n不调用模型补答", "output"),
            ("d", "原回答拆成句子列表 sentences\n若仅一句，则改按双换行拆分", "normal"),
            ("e", "拼接剩余句子 → remainder\nencode → remainder_tokens\ntokens_left = B − len(remainder_tokens)", "normal"),
            ("f", "tokens_left ≥ M？", "decision"),
            ("g", "删除列表末尾一句\nsentences = sentences[:-1]", "normal"),
            ("h", "还有句子？", "decision"),
            ("i", "退出删减循环\n若列表耗尽：代码仍沿用最后一次计算值", "note"),
            ("j", "remainder 追加必要的 </think>\n再追加数学答案前缀或代码前缀\n数学前缀末尾为：\\boxed{", "normal"),
            ("k", "messages 经聊天模板 → question_prompt\n本次模型输入 prompt = question_prompt + remainder", "normal"),
            ("l", "分词、移到设备、调用 model.generate()\nmax_new_tokens=tokens_left；do_sample=True\n切掉本次输入后 decode → final_part", "call"),
            ("m", "full_response = remainder + final_part\n返回字典；forced=True；此函数不保存文件", "output"),
            ("n", "变量边界：full_response 不含 question_prompt\n示例：末尾 \\boxed{ + 新增 402} → \\boxed{402}\n计数细节：tokens_left 在追加标记/前缀之前计算\nresponse_tokens 也沿用追加前的 remainder_tokens", "note"),
        ],
        [("a","b",""),("b","c","是"),("b","d","否"),("d","e",""),("e","f",""),("f","g","否"),("g","h",""),("h","e","是"),("h","i","否"),("f","i","是"),("i","j",""),("j","k",""),("k","l",""),("l","m",""),("m","n","")],
    ),
    "grade-response": (
        "05 · grade_response() 教学判分\n先确认答案位置和格式，再比较数值",
        [
            ("a", "输入：response（已生成的文本）\nexpected（参考答案字符串）", "input"),
            ("b", "包含 </think>？", "decision"),
            ("c", "只取最后一个 </think> 后的文字\n从右侧找最后一个 \\boxed{", "normal"),
            ("d", "找到答案框？", "decision"),
            ("e", "逐字符维护花括号深度 depth\n抽取框内内容，检查括号是否配对", "normal"),
            ("f", "depth 回到 0（答案框闭合）？", "decision"),
            ("g", "predicted = 框内内容去除首尾空白\n预测与参考均符合整数 / 小数 / 分数字面格式？", "decision"),
            ("h", "用 Fraction 转换并精确比较\n转换是否成功？", "decision"),
            ("i", "status=checked\ncorrect = 数值是否相等（True 或 False）\nanswer = predicted", "output"),
            ("r1", "missing_thinking_end", "error"),
            ("r2", "missing_boxed_answer", "error"),
            ("r3", "unfinished_box", "error"),
            ("r4", "unsupported_expression", "error"),
            ("r5", "invalid_numeric_answer", "error"),
            ("r", "以上复核分支均返回\nstatus=needs_review；correct=None\n未能判定，不等于已判错", "note"),
            ("n", "只返回判分字典，不生成、不写文件、不调用外部裁判\n这是教学检查器，不是正式 MATH500 评估器", "note"),
        ],
        [("a","b",""),("b","r1","否"),("b","c","是"),("c","d",""),("d","r2","否"),("d","e","是"),("e","f",""),("f","r3","否"),("f","g","是"),("g","r4","否"),("g","h","是"),("h","r5","抛出转换异常"),("h","i","成功"),("r1","r",""),("r2","r",""),("r3","r",""),("r4","r",""),("r5","r",""),("i","n",""),("r","n","")],
    ),
    "run-grade-inspect": (
        "06 · 教学脚本：run / grade / inspect\n生成记录与判分记录分开保存",
        [
            ("a", "scripts/single_question.py 的子命令\n实际逻辑：src/diagnostic_core.py", "input"),
            ("b", "run\n检查环境、身份、依赖及已有记录\n仅在有待生成阶段时加载模型", "call"),
            ("c", "execute_stage()：阶段文件已存在？", "decision"),
            ("d", "验证记录与依赖哈希\n兼容则 reused；不兼容报错\n不覆盖旧生成", "normal"),
            ("e", "backend.run() 执行该阶段\n组装记录与哈希 → atomic_json\n先保存 base.json / vanilla.json 等", "output"),
            ("f", "grade\n直接进入 grade_run()\n无需模型生成", "call"),
            ("g", "grade_run()\n读取并核验已存在的阶段记录\ngrade_response() 判分 + 记录来源文件哈希\n写入 grading.json，随后打印结果", "output"),
            ("h", "inspect\n进入 inspect_run()\n读取阶段记录及已保存判分", "call"),
            ("i", "显示阶段完成 / 未运行 / 失败 / 无效等状态\n核对判分是否对应当前文件，过期则提示\n不生成、不重新判分、不写结果文件", "normal"),
            ("j", "本图是教学封装：先保存生成，再判分\n官方 inference.py：生成 → 判分 → 写 JSONL\ncompleted 是阶段执行状态，不自动等于答案正确", "note"),
        ],
        [("a","b","run"),("a","f","grade"),("a","h","inspect"),("b","c",""),("c","d","是"),("c","e","否"),("d","g","兼容复用后"),("e","g","保存成功后"),("f","g",""),("h","i",""),("g","j",""),("i","j","")],
    ),
    "regrade-hash": (
        "07 · 重新判分与 SHA-256 复核\n证明哪些文件没变，而不是证明答案正确",
        [
            ("a", "已有 base.json 与 vanilla.json\n先确认终端在项目根目录，路径指向同一运行目录", "input"),
            ("b", "重新判分前：sha256sum 两个生成文件\n把文件路径与哈希保存为 generation-before.sha256", "normal"),
            ("c", "执行 grade --run-dir ...\n读取原生成，重新抽取并判分\n更新 grading.json；不重新调用模型", "call"),
            ("d", "sha256sum -c generation-before.sha256\n重新算当前文件哈希，与清单逐一比较", "normal"),
            ("e", "两个文件均显示 OK？", "decision"),
            ("f", "支持的结论：\n重新判分前后，这两个文件的字节内容一致\n再对比判分状态和答案，检查重试结果", "output"),
            ("g", "先检查路径、文件是否缺失或被修改\n保留错误信息与记录，查清差异", "error"),
            ("h", "OK 不证明模型答案正确，也不证明已异地备份\n答案是否正确看判分；是否上传看远端文件核验\n单次哈希检查本身不证明所有重试逻辑永远无误", "note"),
        ],
        [("a","b",""),("b","c",""),("c","d",""),("d","e",""),("e","f","是"),("e","g","否"),("f","h","")],
    ),
}

COLORS = {"input": ("#E0F2FE", "#0284C7"), "call": ("#EDE9FE", "#8B5CF6"),
          "decision": ("#FEF3C7", "#D97706"), "output": ("#DCFCE7", "#16A34A"),
          "error": ("#FEE2E2", "#DC2626"), "note": ("#F1F5F9", "#64748B"),
          "normal": ("#FFFFFF", "#CBD5E1")}

def quote(value):
    return json.dumps(value, ensure_ascii=False)

def render(name, title, nodes, edges):
    parts = ['digraph G {',
             'graph [rankdir=TB, bgcolor="#F8FAFC", pad=0.35, nodesep=0.32, ranksep=0.35, fontname="PingFang SC", fontsize=25, fontcolor="#0F172A", labelloc=t, label=' + quote(title) + '];',
             'node [shape=box, style="rounded,filled", fontname="PingFang SC", fontsize=17, fontcolor="#0F172A", penwidth=1.4, margin="0.18,0.12"];',
             'edge [color="#64748B", penwidth=1.4, arrowsize=0.7, fontname="PingFang SC", fontsize=14, fontcolor="#475569"];']
    for key, label, kind in nodes:
        fill, stroke = COLORS[kind]
        shape = "diamond" if kind == "decision" else "box"
        parts.append(f'{key} [label={quote(label)}, shape={shape}, fillcolor="{fill}", color="{stroke}"];')
    for a, b, label in edges:
        parts.append(f'{a} -> {b} [label={quote(label)}];')
    parts.append('}')
    dot = ROOT / f"{name}.dot"
    dot.write_text('\n'.join(parts) + '\n')
    for fmt in ("svg", "png"):
        subprocess.run(["dot", f"-T{fmt}", "-Gdpi=130", str(dot), "-o", str(ROOT / f"{name}.{fmt}")], check=True)
    print(name)

if __name__ == "__main__":
    for name, (title, nodes, edges) in GRAPHS.items():
        render(name, title, nodes, edges)
