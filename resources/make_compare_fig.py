# -*- coding: utf-8 -*-
"""生成模型性能对比图（macro-F1 vs 训练耗时），供 README 引用。
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
rows = d["rows"]
names = [r["name"] for r in rows]
f1 = [r["macro_f1"] for r in rows]
fit = [r["fit_s"] for r in rows]

# 颜色：线性模型蓝、树模型橙、贝叶斯绿、kNN 灰、fastText 紫
def color(n):
    if n in ("LinearSVC", "SGD(hinge)", "LogisticRegression"):
        return "#3b82f6"
    if n in ("RandomForest", "ExtraTrees", "XGBoost", "LightGBM"):
        return "#f59e0b"
    if n == "fastText":
        return "#8b5cf6"
    if "NB" in n:
        return "#10b981"
    return "#9ca3af"

BG = "#ffffff"
fig, axes = plt.subplots(1, 2, figsize=(15, 6.4), facecolor=BG)

# 左：macro-F1 横向条形
ax = axes[0]
ax.set_facecolor(BG)
y = list(range(len(names)))[::-1]
bars = ax.barh(y, f1, color=[color(n) for n in names], edgecolor="#334155", linewidth=0.6)
ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xlim(0.55, 0.87)
ax.set_xlabel("macro-F1")
ax.set_title("各模型 macro-F1（同一 TF-IDF 特征 + 无泄漏划分）", fontsize=13, fontweight="bold")
for b, v in zip(bars, f1):
    ax.text(v + 0.003, b.get_y() + b.get_height()/2, f"{v:.4f}",
            va="center", ha="left", fontsize=9, color="#1e293b")
ax.axvline(0.8221, color="#3b82f6", ls="--", lw=1.2, alpha=0.7)
ax.text(0.8221, len(names)-0.3, " 基线 LinearSVC 0.8221", color="#2563eb", fontsize=9)
ax.grid(axis="x", ls=":", alpha=0.4)
ax.set_axisbelow(True)

# 右：训练耗时（对数轴）
ax = axes[1]
ax.set_facecolor(BG)
bars = ax.barh(y, fit, color=[color(n) for n in names], edgecolor="#334155", linewidth=0.6)
ax.set_yticks(y); ax.set_yticklabels(names)
ax.set_xscale("log")
ax.set_xlabel("训练耗时（秒，对数轴）")
ax.set_title("各模型训练耗时", fontsize=13, fontweight="bold")
for b, v in zip(bars, fit):
    ax.text(v * 1.15, b.get_y() + b.get_height()/2, f"{v:.2f}s",
            va="center", ha="left", fontsize=9, color="#1e293b")
ax.grid(axis="x", ls=":", alpha=0.4)
ax.set_axisbelow(True)

fig.suptitle("消费者投诉文本分类 · 11 个分类器横向对比", fontsize=15, fontweight="bold", y=0.99)
fig.text(0.5, 0.005,
         "蓝=线性模型　橙=树模型　紫=fastText　绿=朴素贝叶斯　灰=kNN（同一无泄漏划分；fastText 用同一训练/测试集）",
         ha="center", fontsize=9, color="#64748b")
plt.tight_layout(rect=[0, 0.03, 1, 0.96])

p = os.path.join(OUT, "09_model_compare.png")
fig.savefig(p, dpi=130, facecolor=BG)
print("已写:", p)
