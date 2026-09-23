# -*- coding: utf-8 -*-
"""生成模型性能对比图（macro-F1 vs 耗时），供 README 引用。
输出：resources/figures/09_model_compare.png
"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 11

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "resources")
OUT = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)

d = json.load(open(os.path.join(RES, "tfidf_clf_result.json"), encoding="utf-8"))
# 数据源是各评测脚本"旧行保留 + 新行追加"分批累出来的, 行序等于采集批次,
# 不是排名; 这里统一按 macro-F1 降序再画, 保证图与 README 5.5 的表格同序。
rows = sorted(d["rows"], key=lambda r: -r["macro_f1"])
names = [r["name"] for r in rows]
f1 = [r["macro_f1"] for r in rows]
# 统一耗时口径：本地模型取训练 + 推理；大模型 API 取全量推理墙钟时间
fit = [round((r.get("fit_s") or 0.0) + (r.get("infer_s") or 0.0), 2) for r in rows]


def color(n):
    """按阵营着色：BERT 系（含其压缩产物）一色，传统机器学习及其组合一色，
    第三方大模型 API 一色。"""
    if "LLM" in n or "Jev" in n:
        return "#ef4444"          # 第三方大模型 API 红
    if "BERT" in n or "蒸馏" in n:
        return "#f59e0b"          # BERT 系（微调 / 量化 / 剪枝 / 蒸馏产物）橙
    return "#3b82f6"              # 传统机器学习及其组合 蓝


BG = "#ffffff"
fig, axes = plt.subplots(1, 2, figsize=(16, 7.4), facecolor=BG)

# 左：macro-F1 横向条形
ax = axes[0]
ax.set_facecolor(BG)
y = list(range(len(names)))[::-1]
bars = ax.barh(y, f1, color=[color(n) for n in names], edgecolor="#334155", linewidth=0.6)
ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xlim(0.55, 0.87)
ax.set_xlabel("macro-F1")
ax.set_title("各方法 macro-F1（无泄漏划分）", fontsize=13, fontweight="bold")
for b, v in zip(bars, f1):
    ax.text(v + 0.003, b.get_y() + b.get_height()/2, f"{v:.4f}",
            va="center", ha="left", fontsize=9, color="#1e293b")
best = max(f1)
ax.axvline(best, color="#3b82f6", ls="--", lw=1.2, alpha=0.7)
ax.text(best, len(names)-0.3, f" 最高 {best:.4f}", color="#2563eb", fontsize=9)
ax.grid(axis="x", ls=":", alpha=0.4)
ax.set_axisbelow(True)

# 右：耗时（对数轴）—— 按耗时从短到长独立排序（与左面板 macro-F1 顺序解耦）
ax = axes[1]
ax.set_facecolor(BG)
order_t = sorted(range(len(names)), key=lambda i: fit[i])          # 耗时升序
ypos = [0] * len(names)
for pos, i in enumerate(order_t):
    ypos[i] = len(names) - 1 - pos                                 # 最短的在顶端
bars = ax.barh(ypos, fit,
               color=[color(n) for n in names],
               edgecolor="#334155", linewidth=0.6)
ax.set_yticks(ypos); ax.set_yticklabels(names)
ax.set_xscale("log")
ax.set_xlabel("耗时（秒，对数轴）")
ax.set_title("耗时（本地模型=训练+推理；大模型 API=全量推理，按耗时升序）", fontsize=12, fontweight="bold")
for i, b in enumerate(bars):
    v = fit[i]
    ax.text(v * 1.15, b.get_y() + b.get_height()/2, f"{v:.2f}s",
            va="center", ha="left", fontsize=9, color="#1e293b")
ax.grid(axis="x", ls=":", alpha=0.4)
ax.set_axisbelow(True)

fig.suptitle(f"消费者投诉文本分类 · {len(names)} 个方法横向对比", fontsize=15, fontweight="bold", y=0.99)
fig.text(0.5, 0.005,
         "橙=BERT 系（微调及其压缩产物：量化 / 剪枝 / 蒸馏）　蓝=传统机器学习及其组合（线性 / 贝叶斯 / 树 / kNN / fastText / 集成）　红=第三方大模型 API（同一无泄漏划分）",
         ha="center", fontsize=9, color="#64748b")
plt.tight_layout(rect=[0, 0.03, 1, 0.96])

p = os.path.join(OUT, "09_model_compare.png")
fig.savefig(p, dpi=130, facecolor=BG)
print("已写:", p, f"（{len(names)} 个方法）")
