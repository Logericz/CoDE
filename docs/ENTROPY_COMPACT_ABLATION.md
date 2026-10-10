# 缓存诊断与固定低维特征对照

日期：2026-10-10。用户授权：用已有缓存诊断短长选择器误判，再尝试精简特征；不扩大实验。
本轮是已暴露开发数据上的方法迭代，不是新的确认性验证。旧采集、评分、边界补充、
分析器和结果不改。本文件在新特征结果产生前固定方案，但不声称在观察旧结果前预注册。

## 先诊断什么

固定原20题roster、38个已评分E/T对和27个可追加点。区分：
1. 两侧都错，当前两个分支不能挽救；2. 明明继续可答对却退出；
3. 两侧正确性相同却选择更慢动作。按实测时间加固定错误代价计算动作损失。
仅在诊断时用已知E/T结果定义理想动作，不能把理想选择作为预测或训练输入。

旧g外折重拟合只用于核对保存分数和解释线性贡献：记录训练规模、有效秩、训练拟合、
高度相关特征、标准化极端输入和共有特征重新赋权。相关、系数贡献不是因果证明。

## 固定特征，禁止本轮搜索

| 部件 | 原维度 | 紧凑维度 | 紧凑输入 |
| --- | ---: | ---: | --- |
| f-B：预测追加价值 | 30 | 5 | prefix_tokens、D_short、short confidence、相邻confidence变化、short probability slope |
| f-H：追加熵 | 47 | 7 | f-B加short entropy mean、short entropy slope |
| g_short：短观测决定E/T | 47 | 9 | f-H加short tokens、short ended |
| g_long：长观测决定E/T | 86 | 15 | 完整g_short加long confidence、probability slope、entropy mean/slope、tokens、ended |

相邻confidence变化为当前减上一有效短观测置信度；无上一项时保留缺失，训练折内处理。
slope沿用原曲线摘要中按[0,1]归一化token位置的线性斜率，D沿用保存的短历史值。
概率走势控制已有置信信号的形状，避免只让熵臂看走势；历史变化不由熵替代。
f只处理真正可追加点，其短预算固定21且未结束，不输入这两个常数；g仍使用全部
38个有效配对点训练，必须保留长度和结束状态。long信息只能在付费追加后的g中使用。
不依据错例、系数大小或本轮结果挑选/增删特征，不搜索alpha、阈值、seed或分折。

## 2×2归因对照

执行original/compact g × original/compact f四组，原版为兼容性复现。
每个g、每个lambda只生成一次完整内折训练z和外折测试z；随后两种f及B/H共享
完全相同的标签、顺序、动作选择、常数基线和标签hash。不能因换f重拟合g。

保持原ridge alpha=1、按题4×4折、fold_seed=20261010、最少训练4题、
主lambda=60000ms及30000/120000ms敏感性。标准化和缺失处理只拟合训练折。
短/长g仍独立拟合，本轮只降低特征维度，不同时改共享模型或损失函数。

同一g下比较B/H/常数MAE、MSE、按题区间和相同追加次数效用。
**不同g产生不同z，不能直接用跨g MAE宣布方法改善。** 跨g主要比较相同缓存上的
最终动作正确数、实测剩余时间、错误惩罚和含追加费的总决策损失；已付短probe成本
为共同项，单列为排除项。提供always-E/T与仅作上界的oracle，避免只看预测误差。

所有四组、三个lambda均报告，不选最好一格替代旧主结果。若紧凑版变好，仅能作为
后续方法候选；本批已经用于诊断和设计，不能重新称为未见验证集。不启动新GPU任务。

## 代码与证据

新增独立特征模块`src/entropy_compact_features.py`、对照框架
`src/entropy_feature_ablation.py`及CLI`scripts/analyze_entropy_compact.py`。
复用原数值原语，保持`src/entropy_value_analysis.py`不变。
运行前保存协议/源码/输入hash和固定配置，输出独占新文件。
私有证据统一放在`runs/entropy-value-20261010/selector-compact-001/`，
结果写入`results/entropy-compact-ablation-20261010.md`。
