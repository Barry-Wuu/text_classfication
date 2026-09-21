# -*- coding: utf-8 -*-
"""
消费者投诉文本分类数据集 —— 探索性数据分析（EDA）

用法:
    python eda/eda.py

产出:
    eda/figures/*.png   各类分析图表
    eda/eda_summary.txt 关键统计量汇总

分析维度:
    1. 各类别样本数量分布
    2. 各类别样本长度分布（按字符）
    3. 全部样本长度分布（按字符）
    4. 各类别分句数分布
    5. 全部样本文本长度分布（按分句数）
    6. 原始样本 vs 增强样本 对比

仅依赖标准库 + matplotlib（古法 plt，不用 seaborn）。
字体: Microsoft YaHei（中文正常显示）。
"""
import os
import re
import csv
import statistics
import collections

import matplotlib
matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(ROOT, "train_augmented.csv")
FIG_DIR = os.path.join(HERE, "figures")
SUMMARY_PATH = os.path.join(HERE, "eda_summary.txt")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

# 固定配色（各类别循环取色）
CLR_MAIN = "#4C72B0"
CLR_AUG = "#DD8452"
CLR_ALL = "#55A868"


# ---------------------------------------------------------------------------
# 1. 读取数据
# ---------------------------------------------------------------------------
def load_data(path):
    """返回 list[dict]，每条含 category / text / is_aug / aug_method / variant_idx"""
    rows = []
    with open(path, encoding="utf-8-sig", errors="ignore", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "id": (r.get("id") or "").strip(),
                "category": (r.get("category") or "").strip(),
                "text": r.get("text") or "",
                "is_aug": (r.get("is_aug") or "").strip(),
                "aug_method": (r.get("aug_method") or "").strip(),
                "variant_idx": (r.get("variant_idx") or "").strip(),
            })
    return rows


def count_sentences(text):
    """按中英文句末标点分句，返回非空句数（至少 1）"""
    parts = re.split(r"[。！？!?；;\n]+", text)
    n = len([p for p in parts if p.strip()])
    return max(n, 1)


# ---------------------------------------------------------------------------
# 2. 各类别样本数量
# ---------------------------------------------------------------------------
def fig_category_counts(rows, out):
    total = collections.Counter(r["category"] for r in rows)
    orig = collections.Counter(r["category"] for r in rows if r["is_aug"] == "0")
    cats = [c for c, _ in total.most_common()]

    all_vals = [total[c] for c in cats]
    orig_vals = [orig.get(c, 0) for c in cats]
    aug_vals = [total[c] - orig.get(c, 0) for c in cats]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = range(len(cats))
    ax.bar(x, orig_vals, color=CLR_MAIN, label="原始样本", zorder=3)
    ax.bar(x, aug_vals, bottom=orig_vals, color=CLR_AUG, label="增强样本", zorder=3)
    for i, v in enumerate(all_vals):
        ax.text(i, v + 20, str(v), ha="center", va="bottom", fontsize=8)

    ax.set_title("各类别样本数量分布", fontsize=14)
    ax.set_xlabel("类别", fontsize=11)
    ax.set_ylabel("样本数量", fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.4, zorder=0)
    fig.savefig(out)
    plt.close(fig)
    return dict(total)


# ---------------------------------------------------------------------------
# 3. 各类别样本长度分布（字符数）—— 箱线图
# ---------------------------------------------------------------------------
def fig_category_length_box(rows, out):
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(len(r["text"]))
    cats = sorted(by_cat, key=lambda c: statistics.mean(by_cat[c]))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot([by_cat[c] for c in cats], tick_labels=cats, showfliers=False,
               patch_artist=True,
               boxprops=dict(facecolor=CLR_MAIN, alpha=0.6),
               medianprops=dict(color="red", lw=2))
    ax.set_title("各类别样本长度分布（字符数，已隐藏离群点）", fontsize=14)
    ax.set_xlabel("类别（按平均长度升序）", fontsize=11)
    ax.set_ylabel("文本长度（字符）", fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. 各类别样本长度分布 —— 小提琴图（手工用 KDE 近似，纯 matplotlib）
# ---------------------------------------------------------------------------
def _kde(values, xs, bw=None):
    """极简高斯核密度估计（避免依赖 scipy）"""
    n = len(values)
    if n < 2:
        return [0.0] * len(xs)
    if bw is None:
        sd = statistics.pstdev(values) or 1.0
        bw = 1.06 * sd * (n ** (-1 / 5))
    inv = 1.0 / (n * bw * (2 * 3.141592653589793) ** 0.5)
    out = []
    for x in xs:
        s = 0.0
        for v in values:
            d = (x - v) / bw
            s += 2.718281828459045 ** (-0.5 * d * d)
        out.append(inv * s)
    return out


def fig_category_length_violin(rows, out):
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(len(r["text"]))
    cats = sorted(by_cat, key=lambda c: statistics.median(by_cat[c]))

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, c in enumerate(cats):
        vals = by_cat[c]
        lo, hi = min(vals), max(vals)
        xs = [lo + (hi - lo) * k / 60 for k in range(61)]
        ys = _kde(vals, xs)
        m = max(ys) or 1.0
        ys = [y / m * 0.4 for y in ys]  # 归一化到半宽 0.4
        left = [i - y for y in ys]
        right = [i + y for y in ys]
        ax.fill_betweenx(xs, left, right, color=CLR_ALL, alpha=0.5)
        ax.plot([i, i], [min(vals), max(vals)], color="gray", lw=0.8, alpha=0.6)
        med = statistics.median(vals)
        ax.plot([i - 0.05, i + 0.05], [med, med], color="red", lw=2)

    ax.set_title("各类别样本长度分布（小提琴图，红点为中位数）", fontsize=14)
    ax.set_xlabel("类别（按中位长度升序）", fontsize=11)
    ax.set_ylabel("文本长度（字符）", fontsize=11)
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, rotation=45, ha="right", fontsize=9)
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. 全部样本长度分布（字符数）—— 直方图 + 均值/中位数线
# ---------------------------------------------------------------------------
def fig_all_length_hist(rows, out):
    lens = [len(r["text"]) for r in rows]
    mean = statistics.mean(lens)
    med = statistics.median(lens)

    # 每个整数长度对应一根柱子（不合并区间）
    lo, hi = min(lens), max(lens)
    bins = list(range(lo, hi + 2))

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.hist(lens, bins=bins, color=CLR_ALL, alpha=0.85, edgecolor="white", linewidth=0.3)
    ax.axvline(mean, color="red", ls="--", lw=2, label=f"均值 = {mean:.0f}")
    ax.axvline(med, color="blue", ls="-.", lw=2, label=f"中位数 = {med:.0f}")
    ax.set_title("全部样本长度分布（每字符一根柱）", fontsize=14)
    ax.set_xlabel("文本长度（字符）", fontsize=11)
    ax.set_ylabel("样本数量", fontsize=11)
    ax.set_xlim(lo - 1, hi + 1)
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.savefig(out)
    plt.close(fig)
    return lens


# ---------------------------------------------------------------------------
# 6. 各类别 / 全部分句数分布
# ---------------------------------------------------------------------------
def fig_sentence_counts(rows, out):
    by_cat = collections.defaultdict(list)
    all_cnt = []
    for r in rows:
        n = count_sentences(r["text"])
        by_cat[r["category"]].append(n)
        all_cnt.append(n)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 左：全部样本分句数分布
    cc = collections.Counter(all_cnt)
    xs = sorted(cc)
    ax = axes[0]
    ax.bar(xs, [cc[k] for k in xs], color=CLR_MAIN, zorder=3)
    ax.set_title("全部样本分句数分布", fontsize=13)
    ax.set_xlabel("分句数", fontsize=11)
    ax.set_ylabel("样本数量", fontsize=11)
    ax.grid(axis="y", ls="--", alpha=0.4, zorder=0)

    # 右：各类别平均分句数
    cats = sorted(by_cat, key=lambda c: statistics.mean(by_cat[c]))
    means = [statistics.mean(by_cat[c]) for c in cats]
    ax = axes[1]
    ax.barh(cats, means, color=CLR_AUG, zorder=3)
    for i, v in enumerate(means):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)
    ax.set_title("各类别平均分句数", fontsize=13)
    ax.set_xlabel("平均分句数", fontsize=11)
    ax.grid(axis="x", ls="--", alpha=0.4, zorder=0)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return all_cnt


# ---------------------------------------------------------------------------
# 7. 原始 vs 增强 对比
# ---------------------------------------------------------------------------
def fig_orig_vs_aug(rows, out):
    orig = [len(r["text"]) for r in rows if r["is_aug"] == "0"]
    aug = [len(r["text"]) for r in rows if r["is_aug"] == "1"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # 左：长度分布对比（每个整数长度一根柱子）
    ax = axes[0]
    lo = min(min(orig), min(aug))
    hi = max(max(orig), max(aug))
    bins = list(range(lo, hi + 2))
    ax.hist(orig, bins=bins, color=CLR_MAIN, alpha=0.65, label=f"原始 (n={len(orig)})")
    ax.hist(aug, bins=bins, color=CLR_AUG, alpha=0.65, label=f"增强 (n={len(aug)})")
    ax.set_title("原始 vs 增强：长度分布（每字符一根柱）", fontsize=13)
    ax.set_xlabel("文本长度（字符）", fontsize=11)
    ax.set_ylabel("样本数量", fontsize=11)
    ax.set_xlim(lo - 1, hi + 1)
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.4)

    # 右：关键统计量条形对比
    ax = axes[1]
    stats = {
        "平均长度": [statistics.mean(orig), statistics.mean(aug)],
        "中位长度": [statistics.median(orig), statistics.median(aug)],
        "平均分句数": [statistics.mean([count_sentences(r["text"]) for r in rows if r["is_aug"] == "0"]),
                       statistics.mean([count_sentences(r["text"]) for r in rows if r["is_aug"] == "1"])],
    }
    labels = list(stats)
    x = range(len(labels))
    w = 0.35
    ax.bar([i - w / 2 for i in x], [stats[k][0] for k in labels], w, color=CLR_MAIN, label="原始")
    ax.bar([i + w / 2 for i in x], [stats[k][1] for k in labels], w, color=CLR_AUG, label="增强")
    for i, k in enumerate(labels):
        ax.text(i - w / 2, stats[k][0] + 1, f"{stats[k][0]:.1f}", ha="center", fontsize=8)
        ax.text(i + w / 2, stats[k][1] + 1, f"{stats[k][1]:.1f}", ha="center", fontsize=8)
    ax.set_title("原始 vs 增强：关键统计量", fontsize=13)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend()
    ax.grid(axis="y", ls="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 8. 主流程
# ---------------------------------------------------------------------------
def main():
    rows = load_data(CSV_PATH)
    n = len(rows)
    print(f"读取样本: {n} 条")

    cat_counts = fig_category_counts(rows, os.path.join(FIG_DIR, "01_category_counts.png"))
    fig_category_length_box(rows, os.path.join(FIG_DIR, "02_category_length_box.png"))
    fig_category_length_violin(rows, os.path.join(FIG_DIR, "03_category_length_violin.png"))
    lens = fig_all_length_hist(rows, os.path.join(FIG_DIR, "04_all_length_hist.png"))
    sent_cnt = fig_sentence_counts(rows, os.path.join(FIG_DIR, "05_sentence_counts.png"))
    fig_orig_vs_aug(rows, os.path.join(FIG_DIR, "06_orig_vs_aug.png"))

    # 汇总统计
    orig = [len(r["text"]) for r in rows if r["is_aug"] == "0"]
    aug = [len(r["text"]) for r in rows if r["is_aug"] == "1"]
    lines = []
    lines.append("=" * 60)
    lines.append("消费者投诉文本分类数据集 —— EDA 汇总")
    lines.append("=" * 60)
    lines.append(f"样本总数: {n}")
    lines.append(f"类别数: {len(cat_counts)}")
    lines.append(f"原始样本: {len(orig)}   增强样本: {len(aug)}")
    lines.append("")
    lines.append("【类别样本数量】")
    for c, v in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {c}: {v}")
    lines.append("")
    lines.append("【全部样本长度(字符)】")
    lines.append(f"  最小 {min(lens)}  最大 {max(lens)}  均值 {statistics.mean(lens):.1f}  "
                 f"中位 {statistics.median(lens):.0f}  标准差 {statistics.pstdev(lens):.1f}")
    lines.append("")
    lines.append("【全部样本分句数】")
    lines.append(f"  最小 {min(sent_cnt)}  最大 {max(sent_cnt)}  均值 {statistics.mean(sent_cnt):.2f}  "
                 f"中位 {statistics.median(sent_cnt):.0f}")
    sc = collections.Counter(sent_cnt)
    lines.append("  分布: " + ", ".join(f"{k}句={sc[k]}" for k in sorted(sc)))
    lines.append("")
    lines.append("【各类别长度统计(字符)】")
    by_cat = collections.defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(len(r["text"]))
    for c in sorted(by_cat, key=lambda c: statistics.mean(by_cat[c]), reverse=True):
        v = by_cat[c]
        lines.append(f"  {c}: 均值 {statistics.mean(v):.0f}  中位 {statistics.median(v):.0f}  "
                     f"范围 {min(v)}~{max(v)}")
    lines.append("")
    lines.append("【原始 vs 增强】")
    lines.append(f"  原始: 均值 {statistics.mean(orig):.1f}  中位 {statistics.median(orig):.0f}")
    lines.append(f"  增强: 均值 {statistics.mean(aug):.1f}  中位 {statistics.median(aug):.0f}")

    text = "\n".join(lines)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    print(f"\n图表已输出到: {FIG_DIR}")
    print(f"汇总已输出到: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
