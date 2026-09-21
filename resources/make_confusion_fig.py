# -*- coding: utf-8 -*-
"""四个帕累托最优模型的 18 阶混淆矩阵热力图（每个模型一张）。

配色与标注规则：
  - 白 -> 深蓝（Blues）：0 为纯白，数字越大格子越深蓝；
  - 四张图共用同一色标（同一测试集 1378 条，横向可直接比对）；
  - 采用幂律映射（PowerNorm, gamma<1）提升低值区分辨力 —— 测试集里最小的类
    只有 6 条样本，线性映射下这类格子的非零值会淡到看不见；
  - 每格写出计数；字色按格子实际明度二选一：深格用白字、浅格用黑字。

输入：resources/confusion_matrices.json（三个本地模型）+ resources/bert_confusion.json（BERT）
输出：resources/figures/13_cm_*.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.colors import PowerNorm

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False

DPI = 140
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "resources")
OUT = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)

GAMMA = 0.45          # 幂律指数：<1 抬升低值区，保证小样本类的非零格可见
TEXT_SWITCH = 0.58    # 明度阈值：低于它用白字，否则用黑字
TEXT_DARK = "#0f172a"
TEXT_LIGHT = "#ffffff"

# 帕累托顺序（快 -> 慢），与打榜图一致
ORDER = [
    ("ComplementNB", "13_cm_complementnb.png"),
    ("SGD(hinge)", "14_cm_sgd.png"),
    ("LinearSVC", "15_cm_linearsvc.png"),
    ("BERT微调(抽样1024,ep59)", "16_cm_bert.png"),
]


def load():
    d = json.load(open(os.path.join(RES, "confusion_matrices.json"), encoding="utf-8"))
    labels = d["labels"]
    models = dict(d["models"])
    # BERT 由 Kaggle 内核算出后合并（标签顺序一致，取本地 labels 为准）
    bp = os.path.join(RES, "bert_confusion.json")
    if os.path.exists(bp):
        b = json.load(open(bp, encoding="utf-8"))
        assert b["labels"] == labels, "BERT 类别顺序与本地不一致"
        models["BERT微调(抽样1024,ep59)"] = {
            "matrix": b["matrix"], "acc": b["acc"], "macro_f1": b["macro_f1"],
            "row_sum": b["row_sum"],
        }
        print("已合并 BERT 混淆矩阵: acc=%.4f macroF1=%.4f" % (b["acc"], b["macro_f1"]))
    else:
        print("提示：未找到 bert_confusion.json，仅绘制三个本地模型")
    return labels, models


def luminance(rgb):
    r, g, b = [c * 255.0 for c in rgb[:3]]
    return (0.299 * r + 0.587 * g + 0.114 * b) / 255.0


def draw(name, mat, labels, vmax, acc, f1, out_png):
    n = len(labels)
    fig, ax = plt.subplots(figsize=(9.8, 9.4), dpi=DPI)
    cmap = plt.get_cmap("Blues")
    norm = PowerNorm(gamma=GAMMA, vmin=0.0, vmax=float(vmax))

    ax.imshow(mat, cmap=cmap, norm=norm, origin="upper",
              interpolation="nearest", aspect="equal")

    # 每格写计数；字色按格子明度二选一
    fs = 8.0 if n <= 18 else 6.5
    for i in range(n):
        for j in range(n):
            v = int(mat[i, j])
            col = TEXT_LIGHT if luminance(cmap(norm(v))) < TEXT_SWITCH else TEXT_DARK
            ax.text(j, i, str(v), ha="center", va="center",
                    fontsize=fs, color=col, zorder=3)

    # 网格线：浅灰细线，帮助定位
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="#e2e8f0", linewidth=0.5, alpha=0.7)
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=8.5)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("预测类别", fontsize=11.5, labelpad=8)
    ax.set_ylabel("真实类别", fontsize=11.5, labelpad=8)
    ax.set_title("%s\n准确率 %.4f    macro-F1 %.4f" % (name, acc, f1),
                 fontsize=14, fontweight="bold", pad=12)

    cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.045, pad=0.02)
    cb.set_label("样本数（幂律着色，四图同标尺，上限 %d）" % vmax, fontsize=10)
    cb.ax.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(out_png, dpi=DPI, facecolor="white")
    plt.close(fig)
    print("已写:", os.path.relpath(out_png, ROOT))


def main():
    labels, models = load()
    mats = {}
    for key, _ in ORDER:
        if key in models:
            mats[key] = np.array(models[key]["matrix"], dtype=int)
        else:
            print("缺少模型，跳过:", key)
    vmax = max(int(m.max()) for m in mats.values())
    print("四图共用色标上限 vmax =", vmax)
    for key, fn in ORDER:
        if key not in mats:
            continue
        info = models[key]
        draw(key, mats[key], labels, vmax, info["acc"], info["macro_f1"],
             os.path.join(OUT, fn))
    # 自检：行列和是否等于测试集分布
    if mats:
        key0 = list(mats)[0]
        print("行和自检:", ["OK" if r == c else "MISMATCH"
                        for r, c in zip(mats[key0].sum(axis=1).tolist(),
                                        models[key0]["row_sum"])][:3],
              "... 总和 =", int(mats[key0].sum()))


if __name__ == "__main__":
    main()
