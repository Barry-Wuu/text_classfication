# -*- coding: utf-8 -*-
"""
A/B 对比（速度优先版）：省流标识符对【训练/推理速度】与【内存占用】的影响。

准确率差异（±0.002）是噪声，但省流的真正收益可能在于：
    - TF-IDF 词表更小 → 特征维度更低
    - 稀疏矩阵非零元更少 → 训练/推理更快、内存更省

本脚本对两版数据（clean.csv / clean_ids.csv）在【完全相同】的划分下，
分别测量：
    1. 向量化耗时、词表大小、非零元数、矩阵内存
    2. 模型训练耗时
    3. 推理（预测）耗时
    4. 模型文件大小（pickle）
    5. 准确率（仅作参考，确认没掉）
"""
import os, sys, csv, time, pickle, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ab_test as A   # 复用其加载/分词/划分逻辑

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score, accuracy_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RANDOM_STATE = 42


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return r, time.perf_counter() - t0


def run_one(name, texts, train_idx, test_idx, cats, mk, cache):
    out = {"name": name, "model": mk}

    # 分词（复用缓存）
    key = (name, "tr")
    if key in cache:
        Xtr_tok, Xte_tok = cache[key], cache[(name, "te")]
    else:
        (Xtr_tok, t_tr) = timed(lambda: [" ".join(A.tokenize(texts[i])) for i in train_idx])
        (Xte_tok, t_te) = timed(lambda: [" ".join(A.tokenize(texts[i])) for i in test_idx])
        cache[key], cache[(name, "te")] = Xtr_tok, Xte_tok
        out["tok_train_s"] = t_tr

    ytr = [cats[i] for i in train_idx]
    yte = [cats[i] for i in test_idx]

    # 向量化
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=None)
    Xtr_v, t_fit = timed(lambda: vec.fit_transform(Xtr_tok))
    Xte_v, t_tf = timed(lambda: vec.transform(Xte_tok))

    out["vocab"] = len(vec.vocabulary_)
    out["nnz_train"] = Xtr_v.nnz
    out["nnz_test"] = Xte_v.nnz
    # 稀疏矩阵内存（data+indices+indptr）
    out["mem_mb"] = (Xtr_v.data.nbytes + Xtr_v.indices.nbytes + Xtr_v.indptr.nbytes) / 1024 / 1024
    out["vec_fit_s"] = t_fit
    out["vec_tf_s"] = t_tf

    # 模型
    if mk == "rf":
        clf = RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                     n_jobs=-1, random_state=RANDOM_STATE,
                                     class_weight="balanced_subsample")
    elif mk == "svm":
        clf = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
    else:
        clf = ComplementNB(alpha=0.3)

    _, t_fit_model = timed(lambda: clf.fit(Xtr_v, ytr))
    out["fit_s"] = t_fit_model

    pred, t_pred = timed(lambda: clf.predict(Xte_v))
    out["pred_s"] = t_pred
    out["acc"] = accuracy_score(yte, pred)
    out["macro_f1"] = f1_score(yte, pred, average="macro")

    # 模型文件大小
    out["model_kb"] = len(pickle.dumps(clf)) / 1024
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=["all", "rf", "svm", "nb"])
    args = ap.parse_args()

    meta = A.load_meta()
    cats = [r["category"] for r in meta]
    train_idx, test_idx = A.build_split(meta)
    texts_clean = A.load_text_col(A.CLEAN)
    texts_ids = A.load_text_col(A.CLEAN_IDS)

    models = ["rf", "svm", "nb"] if args.model == "all" else [args.model]
    cache = {}

    rows = []
    for mk in models:
        print(f"===== {mk} =====", flush=True)
        rc = run_one("clean", texts_clean, train_idx, test_idx, cats, mk, cache)
        ri = run_one("clean_ids", texts_ids, train_idx, test_idx, cats, mk, cache)
        for r in (rc, ri):
            rows.append(r)
            print(f"  {r['name']:10s} 词表 {r['vocab']:6d} | nnz {r['nnz_train']:9d} | "
                  f"矩阵 {r['mem_mb']:5.1f}MB | 训练 {r['fit_s']:6.2f}s | 推理 {r['pred_s']*1000:6.1f}ms | "
                  f"模型 {r['model_kb']:6.1f}KB | Acc {r['acc']:.4f}", flush=True)
        print(f"  → 省流: 词表 {(ri['vocab']-rc['vocab'])/rc['vocab']*100:+.2f}% | "
              f"nnz {(ri['nnz_train']-rc['nnz_train'])/rc['nnz_train']*100:+.2f}% | "
              f"训练 {(ri['fit_s']-rc['fit_s'])/rc['fit_s']*100:+.1f}% | "
              f"矩阵内存 {(ri['mem_mb']-rc['mem_mb'])/rc['mem_mb']*100:+.1f}%\n", flush=True)

    print("\n========= 汇总 =========")
    hdr = f"{'模型':6s}{'版本':11s}{'词表':>9s}{'nnz':>11s}{'矩阵MB':>9s}{'训练s':>9s}{'推理ms':>9s}{'模型KB':>9s}{'Acc':>8s}"
    print(hdr)
    for r in rows:
        print(f"{r['model']:6s}{r['name']:11s}{r['vocab']:9d}{r['nnz_train']:11d}{r['mem_mb']:9.1f}"
              f"{r['fit_s']:9.2f}{r['pred_s']*1000:9.1f}{r['model_kb']:9.1f}{r['acc']:8.4f}")


if __name__ == "__main__":
    main()
