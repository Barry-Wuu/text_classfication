# -*- coding: utf-8 -*-
"""
A/B 对比：强标识符省流（clean_ids.csv）是否优于保守清洗（clean.csv）

设计：
    - 严格按「原始样本」做训练/测试划分（原始 6,886 条按 8:2 分层切分），
      增强样本（rule 合成的 23,114 条）只进入训练集，绝不进测试集，
      避免"合成样本泄漏"导致虚高。
    - 测试集 = 原始样本，因为只有原始样本才代表真实分布。
    - 两版数据用【完全相同】的划分（按 id 对齐），保证可比。
    - 三个轻量模型：TF-IDF + 随机森林 / 线性SVM / 朴素贝叶斯。
    - 另输出随机森林的特征重要性 top 词，观察省流前后 token 权重的变化。

用法：
    python eda/ab_test.py            # 跑全部对比（默认随机森林 + SVM + NB）
    python eda/ab_test.py --model rf # 只跑随机森林
"""
import os
import sys
import csv
import argparse
import numpy as np
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
import jieba

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "train_augmented.csv")   # 带 is_aug 标记
CLEAN = os.path.join(ROOT, "clean.csv")
CLEAN_IDS = os.path.join(ROOT, "clean_ids.csv")
RANDOM_STATE = 42


def load_meta():
    """从原始文件取 id / category / is_aug"""
    rows = list(csv.DictReader(open(SRC, encoding="utf-8-sig", errors="ignore", newline="")))
    return rows


def load_text_col(path):
    """读两列 csv → {行序: text}（假定行序与 SRC 一致）"""
    rows = list(csv.reader(open(path, encoding="utf-8")))
    assert rows[0] == ["text", "category"], rows[0]
    return [r[0] for r in rows[1:]]


def tokenize(text):
    return [w for w in jieba.cut(text) if w.strip()]


def build_split(meta):
    """按原始样本分层 8:2 划分；返回 train_idx / test_idx（针对全量行的下标）"""
    from sklearn.model_selection import train_test_split
    orig_idx = [i for i, r in enumerate(meta) if r["is_aug"] == "0"]
    y = [meta[i]["category"] for i in orig_idx]
    tr, te = train_test_split(orig_idx, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    # 训练集 = 原始train + 全部增强
    aug_idx = [i for i, r in enumerate(meta) if r["is_aug"] == "1"]
    train_idx = tr + aug_idx
    return train_idx, te


def evaluate(name, texts, train_idx, test_idx, cats, model_kind, cache=None):
    Xtr = [texts[i] for i in train_idx]
    ytr = [cats[i] for i in train_idx]
    Xte = [texts[i] for i in test_idx]
    yte = [cats[i] for i in test_idx]

    # 预分词（若提供 cache，则复用，避免重复分词）
    key = (name, "tr")
    if cache is not None and key in cache:
        Xtr_tok = cache[key]; Xte_tok = cache[(name, "te")]
    else:
        Xtr_tok = [" ".join(tokenize(t)) for t in Xtr]
        Xte_tok = [" ".join(tokenize(t)) for t in Xte]
        if cache is not None:
            cache[key] = Xtr_tok; cache[(name, "te")] = Xte_tok

    vec = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        max_features=200000,
    )
    Xtr_v = vec.fit_transform(Xtr_tok)
    Xte_v = vec.transform(Xte_tok)

    if model_kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=150, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced_subsample",
        )
    elif model_kind == "svm":
        clf = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
    elif model_kind == "nb":
        clf = ComplementNB(alpha=0.3)
    else:
        raise ValueError(model_kind)

    clf.fit(Xtr_v, ytr)
    pred = clf.predict(Xte_v)
    acc = accuracy_score(yte, pred)
    f1m = f1_score(yte, pred, average="macro")
    f1w = f1_score(yte, pred, average="weighted")
    return {
        "name": name, "model": model_kind,
        "acc": acc, "macro_f1": f1m, "weighted_f1": f1w,
        "n_features": Xtr_v.shape[1],
        "clf": clf, "vec": vec, "yte": yte, "pred": pred,
    }


def top_features(res, k=25):
    clf, vec = res["clf"], res["vec"]
    if not hasattr(clf, "feature_importances_"):
        return []
    names = vec.get_feature_names_out()
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1][:k]
    return [(names[i], imp[i]) for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all", "rf", "svm", "nb"])
    args = ap.parse_args()

    meta = load_meta()
    cats = [r["category"] for r in meta]
    train_idx, test_idx = build_split(meta)
    print(f"样本总数 {len(meta)} | 训练集 {len(train_idx)}（原始 {sum(1 for i in train_idx if meta[i]['is_aug']=='0')} + 增强 {sum(1 for i in train_idx if meta[i]['is_aug']=='1')}）"
          f" | 测试集 {len(test_idx)}（全为原始）\n")

    texts_clean = load_text_col(CLEAN)
    texts_ids = load_text_col(CLEAN_IDS)
    assert len(texts_clean) == len(meta) == len(texts_ids), "行数不一致！"

    models = ["rf", "svm", "nb"] if args.model == "all" else [args.model]
    results = []
    cache = {}
    for mk in models:
        print(f"===== 模型: {mk} =====", flush=True)
        r_clean = evaluate("clean", texts_clean, train_idx, test_idx, cats, mk, cache)
        r_ids = evaluate("clean_ids", texts_ids, train_idx, test_idx, cats, mk, cache)
        for r in (r_clean, r_ids):
            results.append(r)
            print(f"  {r['name']:10s}  特征数 {r['n_features']:6d} | "
                  f"Acc {r['acc']:.4f} | Macro-F1 {r['macro_f1']:.4f} | Weighted-F1 {r['weighted_f1']:.4f}", flush=True)
        d_acc = r_ids["acc"] - r_clean["acc"]
        d_f1 = r_ids["macro_f1"] - r_clean["macro_f1"]
        print(f"  → 省流增益: Acc {d_acc:+.4f} | Macro-F1 {d_f1:+.4f}\n", flush=True)

    # 汇总表
    print("========= 汇总 =========")
    print(f"{'模型':8s} {'版本':10s} {'Acc':>8s} {'MacroF1':>9s} {'WeightedF1':>11s} {'特征数':>8s}")
    for r in results:
        print(f"{r['model']:8s} {r['name']:10s} {r['acc']:8.4f} {r['macro_f1']:9.4f} {r['weighted_f1']:11.4f} {r['n_features']:8d}")

    # 随机森林特征重要性（只对 rf 结果）
    rf_results = [r for r in results if r["model"] == "rf"]
    if rf_results:
        print("\n========= 随机森林 Top-25 特征 =========")
        for r in rf_results:
            print(f"\n[{r['name']}]")
            for w, imp in top_features(r, 25):
                print(f"    {w:20s} {imp:.5f}")
        # 差异：省流后新增/掉出的高权重词
        if len(rf_results) == 2:
            c_set = set(w for w, _ in top_features(rf_results[0], 100))
            i_set = set(w for w, _ in top_features(rf_results[1], 100))
            print("\n仅 clean 版进入 Top100 的词:", "、".join(list(c_set - i_set)[:30]))
            print("仅 clean_ids 版进入 Top100 的词:", "、".join(list(i_set - c_set)[:30]))


if __name__ == "__main__":
    main()
