# -*- coding: utf-8 -*-
"""
消费者投诉文本分类数据集 —— 降维可视化（TF-IDF -> SVD / t-SNE）

用法:
    python eda/dim_reduction.py                 # 全量原始样本
    python eda/dim_reduction.py --sample 2000   # 抽样快速预览

产出:
    eda/figures/09_dim_reduction.png   三面板降维散点图
    eda/dim_reduction.json             画图口径 + kNN 同类率统计

为什么只用原始样本:
    数据集 77% 是规则增强样本, 每条增强样本与源样本几乎是同一句话的变体,
    若一起投影, 同一源样本的若干变体会叠成一个点, 掩盖真实的类间结构。
    因此本图只取 is_aug=0 的原始样本。

三个面板:
    A  TruncatedSVD 前两个主成分 —— 线性投影, 看不清结构是正常的
    B  t-SNE 全 18 类 —— 非线性保局部结构, 能看到真实的簇与咬合
    C  同一个 t-SNE 坐标, 只高亮 5 个互相渗透的类, 其余置灰
       (这 5 类正是 5.10 节标签质量诊断里被点名的互渗簇)

仅依赖标准库 + matplotlib(古法 plt) + sklearn + jieba。
字体: Microsoft YaHei(中文正常显示)。
"""
import os
import json
import time
import argparse
import colorsys

import numpy as np
import pandas as pd
import jieba

import matplotlib
matplotlib.use("Agg")  # 无界面后端
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.lines import Line2D

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# 全局配置
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(ROOT, "train_augmented.csv")
CLASS_PATH = os.path.join(ROOT, "resources", "class.txt")
FIG_PATH = os.path.join(HERE, "figures", "09_dim_reduction.png")
JSON_PATH = os.path.join(HERE, "dim_reduction.json")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

CACHE_DIR = os.path.join(HERE, "_cache")
CACHE_NPZ = os.path.join(CACHE_DIR, "dim_reduction.npz")

SEED = 42
SVD_DIM = 100          # TF-IDF -> SVD 的中间维度(t-SNE 输入)
TSNE_PERP = 30
TSNE_ITER = 1000
KNN_K = 10
LABEL_FS = 9.6         # B 面板簇名标注字号

# 互渗簇: 5.10 节用五判据点名的高重叠类
FOCUS = ["电商平台", "服饰鞋包", "家居日用", "影音娱乐", "数码3C"]
FOCUS_COLORS = {
    "电商平台": "#d62728",   # 红
    "服饰鞋包": "#1f77b4",   # 蓝
    "家居日用": "#2ca02c",   # 绿
    "影音娱乐": "#ff7f0e",   # 橙
    "数码3C": "#9467bd",     # 紫
}


def hsv_palette(n):
    """按黄金角在色环上跳取 n 个高区分度颜色(相邻索引色相差异大, 便于图例对照)"""
    out = []
    for i in range(n):
        h = (i * 0.618033988749895) % 1.0
        out.append(colorsys.hsv_to_rgb(h, 0.62, 0.86))
    return out


def label_cluster_centers(ax, E, y, cats, min_sep=12.0):
    """标注在 t-SNE 平面上"确实分开"的类簇, 并给中央黏合区一段说明。

    判据: 类中心到最近邻类中心的距离 >= min_sep 才标 —— 这是无监督几何,
    不含任何模型。黏在一起的类硬标会糊成一片, 且它们本来就该由 C 面板与
    5.10 节去讨论, 所以统一用一段文字交代。

    文字加白色描边, 保证压在密集散点上仍然可读; annotate 的 xytext 单位是
    点(points), 而几何按像素算, 换算系数 72/dpi 不能漏。
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    to_px = ax.transData.transform

    # 量出单位字符宽高(同字号临时 Text 测一次, 比逐个 draw 快得多)
    probe = fig.text(0.5, 0.5, "示例", fontsize=LABEL_FS, alpha=0.0)
    fig.canvas.draw()
    pb = probe.get_window_extent(renderer=rend)
    ch_w, ch_h = pb.width / 2.0, pb.height
    probe.remove()

    cent = {c: E[y == i].mean(axis=0) for i, c in enumerate(cats)}
    C = np.array([cent[c] for c in cats])
    D = np.hypot(C[:, None, 0] - C[None, :, 0], C[:, None, 1] - C[None, :, 1])
    np.fill_diagonal(D, np.inf)
    dmin = {c: float(D[i].min()) for i, c in enumerate(cats)}

    show = [c for c in cats if dmin[c] >= min_sep]
    hide = [c for c in cats if dmin[c] < min_sep]

    # 簇中心标记：要标的实心描边, 不标的淡一些
    ax.scatter([cent[c][0] for c in show], [cent[c][1] for c in show],
               s=36, facecolor="#111827", edgecolor="white", linewidths=0.9,
               zorder=7)
    ax.scatter([cent[c][0] for c in hide], [cent[c][1] for c in hide],
               s=20, facecolor="#94a3b8", edgecolor="white", linewidths=0.7,
               zorder=6)

    cands = [(0.0, 0.0)] + [(r * np.cos(a), r * np.sin(a))
                            for r in (14.0, 22.0, 32.0, 44.0)
                            for a in np.radians(np.arange(0.0, 360.0, 45.0))]
    anchors = [to_px(cent[c]) for c in cats]
    g = np.array([E[:, 0].mean(), E[:, 1].mean()])
    order = sorted(show, key=lambda c: float(np.hypot(*(cent[c] - g))))

    placed = []
    for c in order:
        P = to_px(cent[c])
        w, h = len(c) * ch_w, ch_h
        best, best_cost = (0.0, 0.0), None
        for dx, dy in cands:
            bx, by = P[0] + dx, P[1] + dy
            r = (bx - w / 2, bx + w / 2, by - h / 2, by + h / 2)
            cost = 0.0
            for q in placed:                        # 与已放标签的重叠面积
                ox = min(r[1], q[1]) - max(r[0], q[0])
                oy = min(r[3], q[3]) - max(r[2], q[2])
                if ox > 0 and oy > 0:
                    cost += ox * oy
            for a in anchors:                       # 别盖住别的类中心
                if abs(bx - a[0]) < w / 2 + 7 and abs(by - a[1]) < h / 2 + 7:
                    cost += 400.0
            cost += 0.25 * float(np.hypot(dx, dy))  # 引线越短越好
            if best_cost is None or cost < best_cost:
                best, best_cost = (dx, dy), cost
        dx, dy = best
        bx, by = P[0] + dx, P[1] + dy
        placed.append((bx - w / 2, bx + w / 2, by - h / 2, by + h / 2))
        ax.annotate(c, cent[c], textcoords="offset points",
                    xytext=(dx * 72.0 / fig.dpi, dy * 72.0 / fig.dpi),
                    fontsize=LABEL_FS, ha="center", va="center",
                    color="#0f172a", zorder=9,
                    path_effects=[patheffects.withStroke(linewidth=2.6,
                                                        foreground="white")],
                    arrowprops=(None if (dx == 0 and dy == 0) else
                                dict(arrowstyle="-", lw=0.85, color="#64748b",
                                     shrinkA=3, shrinkB=5)))

    # 中央黏合区说明(放在图内左上角的空白处, 拆两行避免横向过长)
    n_hide, n_show = len(hide), len(show)
    ax.text(0.015, 0.985,
            f"其余 {n_hide} 个类在中央互相穿插，\n"
            f"各自到最近邻类中心的距离都 < {min_sep:.0f}，簇边界在此基本失效",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.6,
            color="#7c2d12", linespacing=1.65,
            bbox=dict(boxstyle="round,pad=0.42", facecolor="#fff7ed",
                      edgecolor="#fdba74", alpha=0.94))
    return show, hide


def load_ori_texts():
    """读取原始样本(未增强), 返回 texts / labels(整数, 与 class.txt 同序)"""
    with open(CLASS_PATH, encoding="utf-8") as f:
        cats = [ln.strip() for ln in f if ln.strip()]
    idx = {c: i for i, c in enumerate(cats)}

    df = pd.read_csv(CSV_PATH)
    df = df[df["is_aug"] == 0].reset_index(drop=True)
    keep = df["category"].isin(idx)
    df = df[keep].reset_index(drop=True)
    y = df["category"].map(idx).to_numpy()
    return df["text"].astype(str).tolist(), y, cats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0, help="抽样条数, 0 表示全量")
    args = ap.parse_args()

    jieba.setLogLevel(20)
    cut = jieba.lcut
    t_start = time.time()

    texts, y, cats = load_ori_texts()
    if args.sample and args.sample < len(texts):
        rng = np.random.default_rng(SEED)
        sel = rng.choice(len(texts), size=args.sample, replace=False)
        sel.sort()
        texts = [texts[i] for i in sel]
        y = y[sel]
    n = len(texts)
    print(f"原始样本 {n} 条 / {len(cats)} 类", flush=True)

    # ---- 1~4. 特征与降维(带缓存: 调图时不必重跑 t-SNE) ----
    cache_key = f"n{n}_svd{SVD_DIM}_perp{TSNE_PERP}_it{TSNE_ITER}_seed{SEED}"
    cache_ok = False
    if os.path.exists(CACHE_NPZ):
        try:
            z = np.load(CACHE_NPZ, allow_pickle=False)
            cache_ok = str(z["key"]) == cache_key
        except Exception:
            cache_ok = False

    if cache_ok:
        z = np.load(CACHE_NPZ, allow_pickle=False)
        Z, pca2, E = z["Z"], z["pca2"], z["E"]
        evr = z["evr"]
        n_features = int(z["n_features"])
        kl = float(z["kl"])
        print(f"[缓存] 复用 {CACHE_NPZ}（key={cache_key}）", flush=True)
    else:
        t0 = time.time()
        tf = TfidfVectorizer(tokenizer=cut, lowercase=False, token_pattern=None,
                             min_df=2)
        X = tf.fit_transform(texts)
        n_features = int(X.shape[1])
        print(f"TF-IDF {X.shape} 耗时 {time.time()-t0:.1f}s", flush=True)

        t0 = time.time()
        svd = TruncatedSVD(n_components=SVD_DIM, random_state=SEED)
        Z = svd.fit_transform(X)
        evr = svd.explained_variance_ratio_.astype(np.float64)
        print(f"SVD({SVD_DIM}) 耗时 {time.time()-t0:.1f}s | "
              f"前两成分解释方差 {evr[:2].sum():.1%}", flush=True)
        pca2 = Z[:, :2]

        t0 = time.time()
        ts = TSNE(n_components=2, perplexity=TSNE_PERP, init="pca",
                  learning_rate="auto", max_iter=TSNE_ITER, random_state=SEED)
        E = ts.fit_transform(Z)
        kl = float(ts.kl_divergence_)
        print(f"t-SNE 耗时 {time.time()-t0:.1f}s | KL {kl:.2f}", flush=True)

        os.makedirs(CACHE_DIR, exist_ok=True)
        np.savez_compressed(CACHE_NPZ, key=np.array(cache_key), Z=Z, pca2=pca2,
                            E=E, evr=evr, n_features=np.array(n_features),
                            kl=np.array(kl))
        print(f"缓存已写 {CACHE_NPZ}", flush=True)

    # ---- 3. kNN 同类率(在 SVD 空间算, 比原始稀疏空间更稳) ----
    t0 = time.time()
    nn = NearestNeighbors(n_neighbors=KNN_K + 1).fit(Z)
    _, ind = nn.kneighbors(Z)
    ind = ind[:, 1:]                                    # 去掉自己
    same = (y[ind] == y[:, None]).mean(axis=1)          # 每条样本的邻居同类率
    knn_by_cat = {c: float(same[y == i].mean()) for i, c in enumerate(cats)}
    focus_rate = float(np.mean([knn_by_cat[c] for c in FOCUS]))
    other_rate = float(np.mean([v for c, v in knn_by_cat.items() if c not in FOCUS]))
    print(f"kNN(k={KNN_K}) 同类率 整体 {same.mean():.1%} | "
          f"互渗簇 {focus_rate:.1%} | 其余 {other_rate:.1%} 耗时 {time.time()-t0:.1f}s",
          flush=True)

    # ---- 5. 画图: 三面板 ----
    cols = hsv_palette(len(cats))

    # A 面板的离群团: 主成分 2 高于 0.2 的那批(图上与主群之间有明显间隙)
    text_len = np.array([len(t) for t in texts], dtype=float)
    m_out = pca2[:, 1] > 0.2
    n_out = int(m_out.sum())
    len_out = float(np.median(text_len[m_out])) if n_out else 0.0
    len_all = float(np.median(text_len))

    fig, axes = plt.subplots(1, 3, figsize=(17.4, 6.4))

    # A: 线性 SVD 前两主成分
    ax = axes[0]
    for i, c in enumerate(cats):
        m = y == i
        ax.scatter(pca2[m, 0], pca2[m, 1], s=7, c=[cols[i]], alpha=0.62,
                   linewidths=0, label=c)
    ax.set_title("A · 线性投影（SVD 前两个主成分）", fontsize=12.5, pad=9)
    ax.set_xlabel("主成分 1", fontsize=10.5)
    ax.set_ylabel("主成分 2", fontsize=10.5)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.18, linestyle=":")
    ax.text(0.025, 0.975,
            f"左上孤立的 {n_out} 条（{n_out/n:.1%}）是长文本\n"
            f"（中位 {len_out:.0f} 字，全体 {len_all:.0f} 字），与类别无关——\n"
            f"线性投影抓到的是文本长度，不是类别",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.6,
            color="#475569", linespacing=1.65,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white",
                      edgecolor="#cbd5e1", alpha=0.92))

    # B: t-SNE 全类 + 类名标注
    ax = axes[1]
    for i, c in enumerate(cats):
        m = y == i
        ax.scatter(E[m, 0], E[m, 1], s=7, c=[cols[i]], alpha=0.62,
                   linewidths=0)
    ax.set_title("B · t-SNE 全 18 类（非线性保局部结构）", fontsize=12.5, pad=9)
    ax.set_xlabel("t-SNE 维度 1", fontsize=10.5)
    ax.set_ylabel("t-SNE 维度 2", fontsize=10.5)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.18, linestyle=":")
    ax.margins(0.12)
    shown_clusters, merged_clusters = label_cluster_centers(ax, E, y, cats)

    # C: 只高亮互渗簇(与 B 同一套坐标)
    ax = axes[2]
    others = ~np.isin(y, [cats.index(c) for c in FOCUS])
    ax.scatter(E[others, 0], E[others, 1], s=7, c="#cbd5e1", alpha=0.45,
               linewidths=0, zorder=1)
    for c in FOCUS:
        m = y == cats.index(c)
        ax.scatter(E[m, 0], E[m, 1], s=10, c=[FOCUS_COLORS[c]], alpha=0.78,
                   linewidths=0, zorder=2, label=c)
    ax.set_title("C · 只留 5 个互渗类（其余置灰）", fontsize=12.5, pad=9)
    ax.set_xlabel("t-SNE 维度 1", fontsize=10.5)
    ax.set_ylabel("t-SNE 维度 2", fontsize=10.5)
    ax.tick_params(labelsize=9)
    ax.grid(alpha=0.18, linestyle=":")
    ax.legend(markerscale=2.2, fontsize=9.5, loc="upper right",
              framealpha=0.9, ncol=1, handletextpad=0.4, borderpad=0.5)

    # 图例: 18 类横排(仅 B 面板需要, 放在整图底部)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=6.5,
                      markerfacecolor=cols[i], markeredgecolor="none", label=c)
               for i, c in enumerate(cats)]
    fig.legend(handles=handles, loc="lower center", ncol=9, fontsize=9.8,
               frameon=False, bbox_to_anchor=(0.5, -0.055),
               handletextpad=0.35, columnspacing=1.1)

    fig.suptitle(f"降维可视化：TF-IDF → SVD({SVD_DIM}) → t-SNE  "
                 f"（原始样本 {n:,} 条，18 类）",
                 fontsize=14.5, y=1.015)
    fig.text(0.5, -0.135,
             f"读图：A 是线性投影，两轴只承载 "
             f"{evr[:2].sum():.1%} 的方差，类别彼此叠在一起属正常；"
             f"B/C 的 t-SNE 保留的是局部邻域关系，簇分开不代表可分，簇咬合则提示判据接近。\n"
             f"kNN（k={KNN_K}）同类率（在 SVD 空间算）：整体 {same.mean():.1%}，"
             f"互渗簇 5 类 {focus_rate:.1%}，其余 13 类 {other_rate:.1%}"
             f"——簇的咬合不是错觉，互渗簇的邻居确实更容易串到别的类去。",
             ha="center", va="top", fontsize=10.2, color="#334155")

    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(FIG_PATH, facecolor="white")
    plt.close(fig)
    print("图已保存:", FIG_PATH, flush=True)

    # ---- 6. 落盘统计口径 ----
    centroids = {c: [round(float(E[y == i].mean(axis=0)[0]), 2),
                     round(float(E[y == i].mean(axis=0)[1]), 2)]
                 for i, c in enumerate(cats)}
    out = {
        "n_samples": int(n),
        "n_classes": len(cats),
        "only_original": True,
        "tfidf": {"tokenizer": "jieba.lcut", "min_df": 2,
                  "n_features": n_features},
        "svd": {"n_components": SVD_DIM,
                "explained_variance_ratio_top2":
                    [round(float(v), 4) for v in evr[:2]],
                "explained_variance_ratio_sum":
                    round(float(evr.sum()), 4)},
        "tsne": {"perplexity": TSNE_PERP, "max_iter": TSNE_ITER,
                 "random_state": SEED, "kl_divergence": round(kl, 3)},
        "knn": {"k": KNN_K, "space": "SVD100",
                "same_rate_overall": round(float(same.mean()), 4),
                "same_rate_by_category":
                    {c: round(knn_by_cat[c], 4) for c in cats},
                "focus_categories": FOCUS,
                "same_rate_focus_mean": round(focus_rate, 4),
                "same_rate_others_mean": round(other_rate, 4)},
        "linear_projection": {
            "pc2_outlier_threshold": 0.2,
            "n_outlier": n_out,
            "share_outlier": round(n_out / n, 4),
            "median_len_outlier": round(len_out, 1),
            "median_len_all": round(len_all, 1),
            "note": "线性投影的主成分由文本长度主导, 不是类别",
        },
        "tsne_centroids": centroids,
        "tsne_cluster_labeling": {
            "min_sep_for_label": 12.0,
            "rule": "类中心到最近邻类中心的距离 >= 阈值才单独标注, 其余归入中央黏合区",
            "labeled": shown_clusters,
            "merged_in_center": merged_clusters,
        },
        "elapsed_s": round(time.time() - t_start, 1),
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("统计已保存:", JSON_PATH, flush=True)
    print(f"总耗时 {out['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
