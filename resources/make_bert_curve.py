# -*- coding: utf-8 -*-
"""BERT 抽样微调的经典训练曲线（两张）：

  图一 11_bert_loss.png：训练损失 (train loss) 随 epoch
  图二 12_bert_acc.png ：训练集准确率 + 测试集准确率 同图（经典"过拟合张口"图）

数据源：各阶段 result.json 中"从 epoch1 累计"的曲线，取覆盖轮数最多的那份。
"""
import os
import json

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

BG = "#ffffff"
BLUE = "#3b82f6"     # 训练
MAGENTA = "#db2777"  # 测试
GREY = "#64748b"

CAND = ["bert_continue3_result.json", "bert_continue2_result.json",
        "bert_continue_result.json", "bert_epoch_logs.json"]


def load_log():
    best = []
    for fn in CAND:
        p = os.path.join(RES, fn)
        if not os.path.exists(p):
            continue
        j = json.load(open(p, encoding="utf-8"))
        lg = j.get("epoch_log", [])
        if len(lg) > len(best):
            best = lg
    return best


log = load_log()
if not log:
    raise SystemExit("未找到任何 epoch 曲线 json")

ep = [r["epoch"] for r in log]
loss = [r["train_loss"] for r in log]
tr = [r["train_acc"] * 100 for r in log]
te = [r["test_acc"] * 100 for r in log]
f1 = [r["test_macro_f1"] * 100 for r in log]

last = ep[-1]
best_i = max(range(len(te)), key=lambda i: te[i])
f1_i = max(range(len(f1)), key=lambda i: f1[i])
print("轮数 %d（epoch 1~%d）" % (len(ep), last))
print("train_acc %.1f%% -> %.1f%%" % (tr[0], tr[-1]))
print("test_acc  最佳 %.2f%% @epoch%d | 末轮 %.2f%%" % (te[best_i], ep[best_i], te[-1]))
print("test_F1   最佳 %.2f%% @epoch%d" % (f1[f1_i], ep[f1_i]))
print("train-test 末轮差距 %.1f 个点" % (tr[-1] - te[-1]))


def style(ax):
    ax.set_facecolor(BG)
    ax.grid(ls=":", alpha=0.45)
    ax.set_axisbelow(True)
    ax.set_xlabel("epoch")


# ---------------------------------------------------------------- 图一：loss
fig, ax = plt.subplots(figsize=(9.2, 4.9), dpi=150)
fig.patch.set_facecolor(BG)
style(ax)
ax.plot(ep, loss, color="#ef4444", lw=1.9, label="训练损失（当轮抽样 1024 条）")
# 滑动平均，便于看趋势
w = 9
if len(loss) >= w:
    ma = [sum(loss[max(0, i - w + 1):i + 1]) / len(loss[max(0, i - w + 1):i + 1])
          for i in range(len(loss))]
    ax.plot(ep, ma, color="#111827", lw=1.4, ls="--", alpha=0.75,
            label="滑动平均（窗口 %d）" % w)
ax.axvline(last, color=GREY, ls=":", lw=1.2, alpha=0.8, zorder=1)
ax.text(last - 1.5, 2.05, "epoch %d\n过拟合停止" % last, fontsize=9,
        color=GREY, ha="right", va="top")
ax.plot([ep[best_i]], [loss[best_i]], "o", color=MAGENTA, ms=6, zorder=5)
ax.annotate("最佳测试准确率所在轮 epoch %d\nloss=%.4f" % (ep[best_i], loss[best_i]),
            xy=(ep[best_i], loss[best_i]), xytext=(30, 1.15),
            fontsize=9.5, color="#9d174d",
            arrowprops=dict(arrowstyle="->", color="#9d174d", lw=1.1))
ax.set_ylabel("训练损失（交叉熵）")
ax.set_title("BERT 微调训练损失曲线（每轮随机抽 1024 条，共 %d 轮）" % last,
             fontsize=12, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5, framealpha=0.92)
plt.tight_layout()
p1 = os.path.join(OUT, "11_bert_loss.png")
plt.savefig(p1, facecolor=BG)
plt.close(fig)
print("saved", p1)

# ------------------------------------------------------- 图二：train/test acc
fig, ax = plt.subplots(figsize=(9.2, 5.1), dpi=150)
fig.patch.set_facecolor(BG)
style(ax)
ax.plot(ep, tr, color=BLUE, lw=2.0, label="训练集准确率（当轮抽样 1024 条）")
ax.plot(ep, te, color=MAGENTA, lw=2.0, label="测试集准确率（1378 条原始样本）")
ax.fill_between(ep, te, tr, color=BLUE, alpha=0.10)
ax.set_ylabel("准确率（%）")
ax.set_ylim(20, 109)

ax.axvline(last, color=GREY, ls=":", lw=1.2, alpha=0.8, zorder=1)
ax.text(last - 2, 108, "epoch %d 过拟合停止" % last, fontsize=9,
        color=GREY, ha="right", va="top")
ax.plot([ep[best_i]], [te[best_i]], "o", color=MAGENTA, ms=7, zorder=5)
ax.annotate("测试集最高 %.2f%%（epoch %d）\nmacro-F1 %.2f%%"
            % (te[best_i], ep[best_i], f1[best_i]),
            xy=(ep[best_i], te[best_i]),
            xytext=(20, 45),
            fontsize=9.5, color="#9d174d",
            arrowprops=dict(arrowstyle="->", color="#9d174d", lw=1.2))
ax.text(104, 72, "训练-测试差距\n末轮 %.1f 个点" % (tr[-1] - te[-1]),
        fontsize=9.5, color=GREY, ha="right", va="center")

ax.set_title("BERT 微调：训练集 vs 测试集准确率（经典过拟合张口）",
             fontsize=12, fontweight="bold")
ax.legend(loc="lower right", fontsize=9.5, framealpha=0.92)
plt.tight_layout()
p2 = os.path.join(OUT, "12_bert_acc.png")
plt.savefig(p2, facecolor=BG)
plt.close(fig)
print("saved", p2)
