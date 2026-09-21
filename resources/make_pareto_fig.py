# -*- coding: utf-8 -*-
"""打榜风格的帕累托图：横轴=耗时(秒,对数)，纵轴=macro-F1，原点在左下角，
并绘制帕累托最优边界（越靠左上越好：同样耗时下准确率最高、同样准确率下最快）。

输出：resources/figures/10_pareto.png
"""
import os, json
import numpy as np
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


def cost(r):
    """统一到"单条推理耗时"口径，让本地模型与 API 模型可比。

    本地模型：训练耗时 + 推理耗时（fit_s + infer_s）。
    大模型 API：只有一次全测试集推理的墙钟时间 infer_s（无训练概念）。
    为公平起见，横轴统一取"完成整轮评测的墙钟耗时"：
      本地 = fit_s + infer_s； API = infer_s。
    """
    fit = r.get("fit_s") or 0.0
    inf = r.get("infer_s") or 0.0
    return fit + inf


def color(n):
    if n.startswith("Ensemble"):
        return "#06b6d4"          # 集成(专家组) 青
    if n in ("LinearSVC", "SGD(hinge)", "LogisticRegression"):
        return "#3b82f6"          # 线性 蓝
    if n in ("RandomForest", "ExtraTrees", "XGBoost", "LightGBM"):
        return "#f59e0b"          # 树 橙
    if n == "fastText":
        return "#8b5cf6"          # 紫
    if "Jev" in n or "LLM" in n or n.startswith("LLM:"):
        return "#ef4444"          # 大模型 API 红
    if "NB" in n:
        return "#10b981"          # 贝叶斯 绿
    return "#9ca3af"              # kNN 灰


pts = [dict(name=r["name"], t=cost(r), f1=r["macro_f1"]) for r in rows]

# ---- 帕累托最优：耗时越小越好、F1 越大越好 ----
# 一条点 p 被支配 <=> 存在 q 使 q.t <= p.t 且 q.f1 >= p.f1（至少一项严格更好）
pareto = []
for p in pts:
    dominated = False
    for q in pts:
        if q is p:
            continue
        if q["t"] <= p["t"] and q["f1"] >= p["f1"] and (q["t"] < p["t"] or q["f1"] > p["f1"]):
            dominated = True
            break
    if not dominated:
        pareto.append(p)
pareto.sort(key=lambda x: x["t"])

BG = "#ffffff"
fig, ax = plt.subplots(figsize=(11.5, 7.2), facecolor=BG)
ax.set_facecolor(BG)

# 数据点
for p in pts:
    on = any(p is q for q in pareto)
    ax.scatter(p["t"], p["f1"], s=190 if on else 130,
               color=color(p["name"]), edgecolor="#1e293b",
               linewidth=1.3 if on else 0.6, zorder=3)

# 帕累托前沿阶梯线（从左下到右上，先横后竖）
if pareto:
    xs, ys = [], []
    for i, p in enumerate(pareto):
        if i == 0:
            xs.append(p["t"]); ys.append(p["f1"])
        else:
            xs.append(p["t"]); ys.append(pareto[i - 1]["f1"])  # 水平段
            xs.append(p["t"]); ys.append(p["f1"])              # 竖直段
    ax.plot(xs, ys, color="#dc2626", ls="--", lw=1.8, alpha=0.85,
            zorder=2, label="帕累托最优边界")

# 标注：按 x 排序后，相邻点交替上下偏移，减少碰撞
# 标注：按 x 排序；给重叠密集区做多级垂直错位，减少碰撞
def short(n):
    return n.replace("LLM:", "").replace("(cosine)", "")


order = sorted(pts, key=lambda x: x["t"])
for i, p in enumerate(order):
    on = any(p is q for q in pareto)
    if on:
        xytext, va = (10, 6), "bottom"
    else:
        # 以对数耗时分成若干列，同列内交替上下偏移
        lev = i % 4
        dy = [10, -14, 22, -26][lev]
        xytext, va = (9, dy), ("bottom" if dy > 0 else "top")
    ax.annotate(short(p["name"]), (p["t"], p["f1"]),
                textcoords="offset points", xytext=xytext,
                fontsize=9.6 if on else 8.4,
                fontweight="bold" if on else "normal",
                color="#111827" if on else "#475569", zorder=4,
                ha="left", va=va)

ax.set_xscale("log")
ax.set_xlabel("耗时（秒，对数轴；本地 = 训练+推理，大模型 API = 全量推理）",
              fontsize=11)
ax.set_ylabel("macro-F1（无泄漏划分）", fontsize=11)
ax.set_title("打榜视角 · 帕累托最优：耗时 vs 准确率（越靠左上越优）",
             fontsize=14.5, fontweight="bold")

# 原点方向提示（左下角）：优方向是 耗时更小(左) + F1 更大(上) = 左上(↖)
ax.annotate("", xy=(0.015, 0.14), xytext=(0.085, 0.025),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.4))
ax.text(0.09, 0.045, "越靠左上越好\n（更快 · 更准）", transform=ax.transAxes,
        fontsize=9.5, color="#64748b", ha="left", va="bottom")

# 参考线：基线 F1
base_f1 = max(r["macro_f1"] for r in rows)
ax.axhline(base_f1, color="#3b82f6", ls=":", lw=1.1, alpha=0.55)
ax.text(ax.get_xlim()[0], base_f1, f" 最高 macro-F1 = {base_f1:.4f}",
        color="#2563eb", fontsize=9, va="bottom")

ax.grid(True, which="both", ls=":", alpha=0.35)
ax.set_axisbelow(True)
ax.margins(x=0.08, y=0.12)

fig.text(0.5, 0.012,
         "蓝=线性　橙=树模型　紫=fastText　红=大模型 API　绿=朴素贝叶斯　灰=kNN　青=集成(专家组)　"
         "　红点+虚线=帕累托最优前沿（不被任何方法支配）",
         ha="center", fontsize=9, color="#64748b")
plt.tight_layout(rect=[0, 0.03, 1, 1])

p = os.path.join(OUT, "10_pareto.png")
fig.savefig(p, dpi=140, facecolor=BG)
print("已写:", p)
print("帕累托前沿成员:", [q["name"] for q in pareto])
