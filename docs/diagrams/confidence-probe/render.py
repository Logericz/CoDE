"""Source-reading diagrams only; no model execution or inference dependencies."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
renderer = runpy.run_path(str(ROOT.parent / "code-reading" / "render.py"))
render = renderer["render"]
render.__globals__["ROOT"] = ROOT

render("overview", "calcu_max_probs_w_kv()\n一次检查点中的逐 token 试答", [
    ("a", "输入：pred_input_ids、kv_cache、tokenizer\nmethod、tot_step_cap、model", "input"),
    ("b", "设置停止token ID列表\nmethod=0：编码 </think>\nmethod=1：还编码指定的换行/标点字符串", "normal"),
    ("c", "初始化：last_token=-1，n=0，累计值S=0\n两个空列表：pred_tokens、token_probs\n有输入KV则深拷贝并移到输入设备；否则为None", "normal"),
    ("d", "last_token 已在 stop_ids 中？", "decision"),
    ("e", "选择本轮模型输入\n首次：pred_input_ids + 当前KV\n随后：上次生成的单个token + 当前KV", "normal"),
    ("f", "model(...) 前向计算，并更新本次试答的KV\n取最后位置 logits → softmax → torch.max\n选出 max_index 与 max_value（贪心选取）", "call"),
    ("g", "先累计，再更新 last_token\n此前last_token=-1：S不变\n其他轮：method=0加log(p)，method=1加p", "normal"),
    ("h", "每一轮都保存 token ID 与概率\nlast_token = max_index\nn = n + 1", "normal"),
    ("i", "n > tot_step_cap？", "decision"),
    ("j", "退出循环\nended_with_think = (last_token in stop_ids)\n按现有公式计算最终分数（见算例图）", "normal"),
    ("k", "把生成token列表decode成 predicted_answer\n释放本次试答的临时KV引用", "normal"),
    ("l", "返回字典：total_prob_max、predicted_answer\ntoken_ids、token_probs、ended_with_think", "output"),
    ("m", "本函数不比较早停阈值，也不写实验文件\n调用者 method_deer() 使用分数作停止判断\ncap=20允许第21步；停止标记可能更早结束循环", "note"),
], [("a","b",""),("b","c",""),("c","d",""),("d","e","否"),("e","f",""),("f","g",""),("g","h",""),("h","i",""),("i","d","否，继续"),("i","j","是"),("d","j","是"),("j","k",""),("k","l",""),("l","m","")])

render("score-example", "保存的概率 ≠ 最终参加评分的概率\nmethod=0 的源码算例（纯教学，未运行模型）", [
    ("a", "假设依次选出三个token：A、B、停止token\n对应概率：p1=0.70，p2=0.81，p3=0.64", "input"),
    ("b", "保存路径\ntoken_probs 完整保存 [0.70, 0.81, 0.64]", "normal"),
    ("c", "累计路径\n第1步：S=0（首token未加入）\n第2步：S=log(0.81)\n第3步：S=log(0.81)+log(0.64)", "normal"),
    ("d", "循环后：无论停止原因，减去最后概率的log\n分子 S − log(p3) = log(0.81)\n分母 n − 1 = 2", "call"),
    ("e", "返回分数 exp(log(0.81)/2) = 0.90\n实际保留1个概率项，但分母是2\n不能解释成保留项的普通几何平均", "output"),
    ("f", "继续看短试答边界（method=0）", "normal"),
    ("g", "仅生成1个token\n分母 n−1 = 0\np1在(0,1)时可能为inf；p1=1时为NaN", "error"),
    ("h", "仅生成2个token，且概率为正\n分子 log(p2)−log(p2) = 0\n分数 exp(0/1) = 1，不代表答案可靠", "error"),
    ("i", "教学封装的 _guard_probe()\n先记录试答，再拒绝 ≤2 token 的结果\n也拒绝非有限值等异常，保留诊断", "note"),
    ("j", "源码行为说明：末token在达到步数上限时\n也会被减去，并不保证它是停止token\n以上分数不是答案正确率，也不是GPU实验结果", "note"),
], [("a","b","所有概率都保存"),("a","c","部分概率参加计算"),("c","d",""),("d","e",""),("b","e","对照"),("e","f",""),("f","g","n=1"),("f","h","n=2"),("g","i",""),("h","i",""),("i","j","")])
