# -*- coding: utf-8 -*-
"""各类标签的分类准确率 与 各类样本量 的关系分析。

用法:
    python resources/make_size_accuracy_fig.py

产出:
    resources/figures/20_size_vs_accuracy.png   三面板散点图
    resources/size_vs_accuracy.json             相关分析的全部数字

要回答的问题:
    某一类之所以分不准, 是因为它样本太少(数据量问题), 还是因为它和别的类
    本来就分不开(标签定义问题)?

做法:
    把 18 个类各当成一个观测点, 用"4 个模型的平均逐类召回率"当因变量, 分别与
    ① 类别样本数(全量 / 测试集)  ② 无监督标签质量指标(最近类中心自洽率、
    SVD 空间 kNN 同类率) 求相关, 看谁解释得动。

口径说明(重要):
    - 召回率来自同一份无泄漏测试集(1,378 条)上的四个模型: ComplementNB、
      SGD(hinge)、LinearSVC、BERT 微调(epoch 59)。
    - "集体错率"与召回率同源(都是模型在测试集上的表现), 它对召回的高相关属于
      同义反复, 本脚本只把它当"两套口径是否自洽"的检查, 不作为独立证据。
    - 母婴食品测试集仅 6 条, 逐类均值极不稳, 相关分析剔除样本数 < 10 的类,
      并在图中单独标注。
"""
import os
import json
import colorsys

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patheffects
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG_PATH = os.path.join(HERE, "figures", "20_size_vs_accuracy.png")
JSON_PATH = os.path.join(HERE, "size_vs_accuracy.json")

DPI = 130
LABEL_FS = 8.6
MIN_TEST_N = 10          # 相关分析里剔除测试样本过少的类

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = DPI
plt.rcParams["savefig.bbox"] = "tight"

C_MAIN = "#2563eb"       # 普通点
C_FOCUS = "#dc2626"      # 标注出的极端类
C_FIT = "#64748b"        # 回归线


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def load_all():
    cats = [l.strip() for l in open(os.path.join(ROOT, "resources", "class.txt"),
                                    encoding="utf-8") if l.strip()]
    cm = json.load(open(os.path.join(HERE, "confusion_matrices.json"), encoding="utf-8"))
    bc = json.load(open(os.path.join(HERE, "bert_confusion.json"), encoding="utf-8"))
    mats = [np.array(cm["models"]["ComplementNB"]["matrix"]),
            np.array(cm["models"]["SGD(hinge)"]["matrix"]),
            np.array(cm["models"]["LinearSVC"]["matrix"]),
            np.array(bc["matrix"])]
    rec = np.array([m.diagonal() / m.sum(axis=1) for m in mats])      # 4 x 18
    mean_rec = rec.mean(axis=0)

    lm = pd.read_csv(os.path.join(HERE, "label_map.csv"))
    full_n = np.array([int(lm.loc[lm["name"] == c, "count"].iloc[0]) for c in cats])
    test_n = np.array(cm["split"]["test_dist"], dtype=float)

    q = json.load(open(os.path.join(HERE, "label_quality_diag.json"), encoding="utf-8"))
    self_rate = np.array([q["self_rate"][c] for c in cats])
    collective = np.array([q["collective_wrong_rate"][c] for c in cats])
    dr = json.load(open(os.path.join(ROOT, "eda", "dim_reduction.json"), encoding="utf-8"))
    knn = np.array([dr["knn"]["same_rate_by_category"][c] for c in cats])

    return dict(cats=cats, rec=rec, mean_rec=mean_rec, full_n=full_n,
                test_n=test_n, self_rate=self_rate, collective=collective,
                knn=knn, model_names=list(cm["models"].keys()) + ["BERT微调"])


# ---------------------------------------------------------------------------
# 散点标注: 候选方位贪心避让 (与 eda/dim_reduction.py 同一套做法)
# ---------------------------------------------------------------------------
def label_points(ax, xs, ys, names, highlight=(), fs=LABEL_FS):
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    to_px = ax.transData.transform

    probe = fig.text(0.5, 0.5, "示例", fontsize=fs, alpha=0.0)
    fig.canvas.draw()
    pb = probe.get_window_extent(renderer=rend)
    ch_w, ch_h = pb.width / 2.0, pb.height
    probe.remove()

    cands = [(0.0, 0.0)] + [(r * np.cos(a), r * np.sin(a))
                            for r in (11.0, 17.0, 25.0, 35.0)
                            for a in np.radians(np.arange(0.0, 360.0, 45.0))]
    anchors = [to_px((x, y)) for x, y in zip(xs, ys)]
    placed = []
    order = sorted(range(len(names)), key=lambda i: names[i] not in highlight)
    for i in order:
        P = to_px((xs[i], ys[i]))
        w, h = len(names[i]) * ch_w, ch_h
        best, best_cost = (0.0, 0.0), None
        for dx, dy in cands:
            bx, by = P[0] + dx, P[1] + dy
            r = (bx - w / 2, bx + w / 2, by - h / 2, by + h / 2)
            cost = 0.0
            for q in placed:
                ox = min(r[1], q[1]) - max(r[0], q[0])
                oy = min(r[3], q[3]) - max(r[2], q[2])
                if ox > 0 and oy > 0:
                    cost += ox * oy
            for a in anchors:
                if abs(bx - a[0]) < w / 2 + 6 and abs(by - a[1]) < h / 2 + 6:
                    cost += 300.0
            cost += 0.25 * float(np.hypot(dx, dy))
            if best_cost is None or cost < best_cost:
                best, best_cost = (dx, dy), cost
        dx, dy = best
        bx, by = P[0] + dx, P[1] + dy
        placed.append((bx - w / 2, bx + w / 2, by - h / 2, by + h / 2))
        ax.annotate(names[i], (xs[i], ys[i]), textcoords="offset points",
                    xytext=(dx * 72.0 / fig.dpi, dy * 72.0 / fig.dpi),
                    fontsize=fs, ha="center", va="center", zorder=9,
                    color=C_FOCUS if names[i] in highlight else "#0f172a",
                    fontweight="bold" if names[i] in highlight else "normal",
                    path_effects=[patheffects.withStroke(linewidth=2.4,
                                                         foreground="white")])


def panel(ax, x, y, title, xlabel, ylabel, mask=None, hl=(),
          note=None, note_color="#dc2626", show_ylabel=True):
    """画一个散点面板。

    mask: 参与相关分析与拟合的样本(True); 不在其中的点仍绘出但用灰色空心,
    并明确标注"未纳入统计" —— 图上的 r 与脚本正文必须同一口径。
    """
    n = len(x)
    if mask is None:
        mask = np.ones(n, dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    ax.scatter(x[~mask], y[~mask], s=46, facecolor="none", edgecolor="#94a3b8",
               linewidths=1.2, alpha=0.9, zorder=5)
    ax.scatter(x[mask], y[mask], s=52, facecolor=C_MAIN, edgecolor="white",
               linewidths=1.0, alpha=0.85, zorder=5)

    lr = stats.linregress(x[mask], y[mask])
    xs = np.linspace(x[mask].min(), x[mask].max(), 50)
    ax.plot(xs, lr.intercept + lr.slope * xs, ls="--", lw=1.5,
            color=C_FIT, zorder=4)
    pr, pp = stats.pearsonr(x[mask], y[mask])
    sr, sp = stats.spearmanr(x[mask], y[mask])

    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel(xlabel, fontsize=10.5)
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=10.5)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.2, ls=":")
    ax.margins(x=0.18, y=0.26)

    txt = (f"n = {int(mask.sum())} 类\n"
           f"Pearson r = {pr:+.3f}（p = {pp:.3f}）\n"
           f"Spearman ρ = {sr:+.3f}（p = {sp:.3f}）")
    if note:
        txt += "\n" + note
    ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=8.6, color="#0f172a", linespacing=1.55,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="#f8fafc",
                      edgecolor=note_color, alpha=0.95))
    if (~mask).any():
        ax.scatter([], [], s=46, facecolor="none", edgecolor="#94a3b8",
                   linewidths=1.2, label="测试样本 < 10，未纳入统计")
        ax.legend(loc="lower right", fontsize=8.2, framealpha=0.92,
                  handletextpad=0.4, borderpad=0.4)
    return pr, sr


def main():
    D = load_all()
    cats, rec, mean_rec = D["cats"], D["rec"], D["mean_rec"]
    full_n, test_n = D["full_n"], D["test_n"]
    keep = test_n >= MIN_TEST_N
    kept = [c for c, k in zip(cats, keep) if k]
    print("参与相关分析的类: %d 个（剔除测试样本 < %d 的: %s）"
          % (keep.sum(), MIN_TEST_N, [c for c, k in zip(cats, keep) if not k]))

    acc = mean_rec * 100
    ks = np.array(keep)
    stats_out = {}
    for name, arr in (("类别样本数", full_n),
                      ("训练集样本数", None),
                      ("测试集样本数", test_n),
                      ("无模型自洽率", D["self_rate"]),
                      ("kNN同类率", D["knn"]),
                      ("集体错率", -D["collective"])):
        if arr is None:
            continue
        pr, pp = stats.pearsonr(np.asarray(arr)[ks], acc[ks])
        sr, sp = stats.spearmanr(np.asarray(arr)[ks], acc[ks])
        stats_out[name] = {"pearson_r": round(float(pr), 3), "pearson_p": round(float(pp), 4),
                           "spearman_rho": round(float(sr), 3), "spearman_p": round(float(sp), 4)}
        print(f"  {name:<12} Pearson r={pr:+.3f} (p={pp:.3f})  Spearman ρ={sr:+.3f} (p={sp:.3f})")

    # 配对: 先看样本数完全相同、其次最接近的两个类, 召回相差多少
    diff_pairs = []
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            diff_pairs.append((abs(full_n[i] - full_n[j]), -abs(acc[i] - acc[j]),
                               cats[i], cats[j], int(full_n[i]), acc[i], acc[j]))
    diff_pairs.sort()
    n_gap, neg_rgap, ca, cb, n_ab, acc_a, acc_b = diff_pairs[0]
    gap = -neg_rgap
    print(f"\n配对极值: {ca}({n_ab} 条, {acc_a:.1f}%) vs {cb}({int(full_n[cats.index(cb)])} 条,"
          f" {acc_b:.1f}%) -> 样本数差 {n_gap} 条, 召回差 {gap:.1f} 个点")

    # ---------------- 画图 ----------------
    fig, axes = plt.subplots(1, 3, figsize=(17.6, 5.9), facecolor="white")
    hl = ("电商平台", "影音娱乐", "金融支付", "通讯运营商", "母婴食品")

    # A: 样本数 vs 准确率
    ax = axes[0]
    panel(ax, full_n, acc, "A · 类别样本数 vs 平均召回", "类别样本数（全量）",
          "四模型平均召回（%）", mask=keep,
          note="r 接近 0：样本量解释不了准确率")
    label_points(ax, full_n, acc, cats, highlight=hl)
    # 把"样本数相同、召回却差近 40 点"的配对连起来并注明
    ia, ib = cats.index(ca), cats.index(cb)
    ax.annotate("", xy=(full_n[ib], acc[ib]), xytext=(full_n[ia], acc[ia]),
                arrowprops=dict(arrowstyle="<->", color="#dc2626", lw=1.6,
                                ls=(0, (5, 3)), alpha=0.85), zorder=6)
    ax.text((full_n[ia] + full_n[ib]) / 2 + 12, (acc[ia] + acc[ib]) / 2,
            f"{ca} 与 {cb}\n同为 {n_ab} 条，差 {gap:.1f} 个点",
            fontsize=9, color="#b91c1c", ha="left", va="center", linespacing=1.6,
            path_effects=[patheffects.withStroke(linewidth=2.6, foreground="white")])

    # B: 无模型自洽率 vs 准确率
    ax = axes[1]
    panel(ax, D["self_rate"], acc, "B · 无模型自洽率 vs 平均召回",
          "最近类中心自洽率（无监督）", "四模型平均召回（%）", mask=keep,
          note="r 高：标签可分性才是主因", note_color="#15803d")
    label_points(ax, D["self_rate"], acc, cats, highlight=hl)

    # C: kNN 同类率 vs 准确率
    ax = axes[2]
    panel(ax, D["knn"], acc, "C · kNN 同类率 vs 平均召回",
          "kNN（k=10）同类率（无监督，SVD 空间）", "四模型平均召回（%）", mask=keep,
          note="两个独立指标都指向同一结论", note_color="#15803d")
    label_points(ax, D["knn"], acc, cats, highlight=hl)

    fig.suptitle("各类样本量 与 分类准确率 的关系：样本多不等于分得准",
                 fontsize=14.5, y=1.02)
    fig.text(0.5, -0.135,
             "读图：每点一个类，纵轴都是同一份无泄漏测试集上四个模型"
             "（ComplementNB / SGD(hinge) / LinearSVC / BERT 微调）的平均逐类召回；"
             "虚线是线性拟合。\n"
             "A 的样本数与准确率基本无关（r 接近 0），B、C 两个"
             "不含任何模型的无监督指标却和准确率高度一致 —— "
             "瓶颈在标签可分性，不在数据量。\n"
             "母婴食品测试集只有 6 条样本，逐类均值不稳，相关分析已将其剔除，"
             "图中仍按实际值绘出仅供参考。",
             ha="center", va="top", fontsize=10, color="#334155")

    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(FIG_PATH, facecolor="white")
    plt.close(fig)
    print("\n图已保存:", FIG_PATH)

    out = {
        "n_classes": len(cats),
        "n_classes_in_corr": int(keep.sum()),
        "excluded_low_test_n": [c for c, k in zip(cats, keep) if not k],
        "recall_source": "ComplementNB / SGD(hinge) / LinearSVC / BERT微调(ep59) "
                         "在同一无泄漏测试集(1378 条)上的逐类召回均值",
        "correlations": stats_out,
        "note_collective": "集体错率与召回率同源(都是同一批模型在测试集上的表现), "
                           "对本项的高相关属同义反复, 仅作口径自洽性检查, 不作独立证据",
        "pair_contrast": {"a": ca, "b": cb, "same_n": n_ab,
                          "acc_a": round(float(acc_a), 1), "acc_b": round(float(acc_b), 1),
                          "gap_points": round(float(gap), 1)},
        "per_class": {c: {"full_n": int(full_n[i]), "test_n": int(test_n[i]),
                          "mean_recall": round(float(mean_rec[i]), 4),
                          "self_rate": round(float(D["self_rate"][i]), 4),
                          "knn_same_rate": round(float(D["knn"][i]), 4),
                          "collective_wrong": round(float(D["collective"][i]), 4)}
                      for i, c in enumerate(cats)},
    }
    json.dump(out, open(JSON_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("统计已保存:", JSON_PATH)


if __name__ == "__main__":
    main()
