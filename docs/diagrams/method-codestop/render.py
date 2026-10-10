"""Render the audited CoDE study overview; no model execution."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
render = runpy.run_path(str(ROOT.parent / "code-reading" / "render.py"))["render"]
render.__globals__["ROOT"] = ROOT
render("overview", "CoDE-Stop 主函数：检查、停止、回退与返回\n源码复习图 · 2026-10-10", [
    ("a", "输入：模型、分词器、题目、已有回答与方法参数", "input"),
    ("b", "构造题目与答案前缀，定位旧回答的候选检查点\nstop_word 编码后的各 token ID + EOS ID", "normal"),
    ("c", "还有检查点？", "decision"),
    ("d", "准备本次试答\ncp_cache=True：更新前缀KV，传试答前缀与缓存\nFalse：完整题目 + 保留推理 + 试答前缀", "normal"),
    ("e", "调用 calcu_max_probs_w_kv()\n得到试答、概率、分数、结束标记；加入 prob_checks", "call"),
    ("f", "计算退化分数与有效置信度阈值\n不足3个检查点：退化分数为0\n启用ramp：阈值使用检查点序号i，否则固定阈值", "normal"),
    ("g", "两条停止分支取或\n置信度分支：(不要求ewt 或结束标记满足) 且分数 > 有效阈值\n退化分支：退化分数 > 退化阈值（独立于ewt）", "call"),
    ("h", "任一分支满足？", "decision"),
    ("i", "选定答案位置\nrollback=True：已检查记录中置信度最高的位置\nFalse：当前检查位置", "normal"),
    ("j", "拼正式生成输入：题目 + 选定位置前推理 + 正式前缀\n回退到不同位置：不传当前检查点KV\n未实际回退：按cp_cache决定传KV或None", "normal"),
    ("k", "另行调用 model.generate() 生成最终答案\n切掉题目后decode；统计 response_tokens 与 work", "call"),
    ("l", "返回字典：stopped_early=True\nresponse、prob_checks、阈值、计数、原回答、rollback_used", "output"),
    ("m", "返回原回答与检查记录：stopped_early=False\nwork = 原回答重编码长度 + 全部试答token数\n也覆盖没有检查点的情况", "output"),
    ("n", "函数只返回；外层负责判分与保存\n官方 inference.py：生成 → 判分 → JSONL\n教学封装的文件格式与保存顺序另行核对", "note"),
    ("o", "早停 work 用当前检查位置，不用回退后位置\n再加正式前缀、新生成答案、全部试答token\n这是实现计数，不等于完整计算量或实测耗时", "note"),
], [("a","b",""),("b","c",""),("c","d","有"),("d","e",""),("e","f",""),("f","g",""),("g","h",""),("h","c","否，下一检查点"),("h","i","是"),("i","j",""),("j","k",""),("k","l",""),("c","m","无"),("l","n",""),("m","n",""),("n","o","")])
