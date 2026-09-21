# -*- coding: utf-8 -*-
"""TF-IDF + 多算法对比（本机 CPU，无泄漏划分）。

沿用 compare_no_leak.py 的评估协议：
  - 按源 id 分组划分：测试 = 原始样本中切出的 20% 源；其全部增强变体不进训练
  - 训练 = 其余原始 + 非测试源的增强；测试只用原始样本
在同一份 TF-IDF 特征上横向比多个分类器：
  LinearSVC / LogisticRegression / RandomForest / ExtraTrees / ComplementNB /
  MultinomialNB / SGD(hinge) / kNN(cosine)
用法：python compare_tfidf_clf.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import jieba
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "train_augmented.csv")
SEED = 42
MAX_LEN = 200  # 仅记录，特征用 TF-IDF 不涉及

jieba.setLogLevel(20)
cut = lambda t: jieba.lcut(t)


def build_split(df):
    src = df["id"].astype(str).str.split("#").str[0]
    is_aug = df["is_aug"].to_numpy()
    ori_idx = np.where(is_aug == 0)[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    ori_all = np.arange(len(df))
    tr_ori, te_ori = next(gss.split(ori_all[ori_idx], groups=src.to_numpy()[ori_idx]))
    tr_ori = ori_idx[tr_ori]; te_ori = ori_idx[te_ori]
    test_src = set(src.to_numpy()[te_ori])
    aug_idx = np.where(is_aug == 1)[0]
    train_aug_idx = np.array([i for i in aug_idx if src.to_numpy()[i] not in test_src])
    train = np.concatenate([tr_ori, train_aug_idx])
    return train, te_ori, test_src


def main():
    df = pd.read_csv(CSV)
    cats = sorted(df["category"].unique())
    m = {c: i for i, c in enumerate(cats)}
    y = df["category"].map(m).to_numpy()
    texts = df["text"].astype(str).tolist()
    train, test, test_src = build_split(df)
    ytr, yte = y[train], y[test]
    n_tr_aug = int((df["is_aug"].to_numpy()[train] == 1).sum())
    print(f"训练 {len(train)}（原始 {len(train)-n_tr_aug} + 增强 {n_tr_aug}）/ "
          f"测试 {len(test)}（全原始）；测试源 {len(test_src)} 个", flush=True)

    # ---- 共享 TF-IDF 特征 ----
    t0 = time.time()
    tfidf = TfidfVectorizer(tokenizer=cut, lowercase=False, token_pattern=None, min_df=2)
    Xtr = tfidf.fit_transform([texts[i] for i in train])
    Xte = tfidf.transform([texts[i] for i in test])
    print(f"TF-IDF 特征：dim={Xtr.shape[1]} nnz={Xtr.nnz} 用时 {time.time()-t0:.1f}s", flush=True)

    Xtr_d = Xtr
    Xte_d = Xte

    # 稠密版（树模型/kNN 用）
    t0 = time.time()
    Xtr_dense = Xtr.toarray().astype("float32")
    Xte_dense = Xte.toarray().astype("float32")
    print(f"转稠密：{Xtr_dense.shape} 用时 {time.time()-t0:.1f}s", flush=True)

    models = [
        ("LinearSVC",          LinearSVC(C=1.0),                       "sparse"),
        ("LogisticRegression", LogisticRegression(max_iter=1000),      "sparse"),
        ("SGD(hinge)",         SGDClassifier(loss="hinge", random_state=SEED), "sparse"),
        ("ComplementNB",       ComplementNB(),                         "sparse"),
        ("MultinomialNB",      MultinomialNB(),                        "sparse"),
        ("RandomForest",       RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED), "dense"),
        ("ExtraTrees",         ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),   "dense"),
        ("kNN(cosine)",        KNeighborsClassifier(n_neighbors=15, metric="cosine", n_jobs=-1),        "dense"),
    ]

    rows = []
    for name, clf, kind in models:
        Xa = Xtr if kind == "sparse" else Xtr_dense
        Xb = Xte if kind == "sparse" else Xte_dense
        try:
            t0 = time.time()
            clf.fit(Xa, ytr)
            fit = time.time() - t0
            t0 = time.time()
            pred = clf.predict(Xb)
            inf = time.time() - t0
            f1 = f1_score(yte, pred, average="macro")
            acc = accuracy_score(yte, pred)
            print(f"  {name:20s} macroF1={f1:.4f} acc={acc:.4f} "
                  f"训练={fit:.1f}s 推理={inf:.1f}s", flush=True)
            rows.append(dict(name=name, macro_f1=round(float(f1), 4),
                             acc=round(float(acc), 4), fit_s=round(fit, 2),
                             infer_s=round(inf, 2)))
        except Exception as e:
            print(f"  {name:20s} 失败: {e!r}", flush=True)

    rows.sort(key=lambda r: -r["macro_f1"])
    print("\n=== 排名 ===")
    for r in rows:
        print(f"  {r['name']:20s} F1={r['macro_f1']:.4f} acc={r['acc']:.4f}")

    out = {"rows": rows, "best": rows[0] if rows else None,
           "split": {"train": int(len(train)), "test": int(len(test)),
                     "test_sources": int(len(test_src)), "feat_dim": int(Xtr.shape[1])}}
    p = os.path.join(ROOT, "resources", "tfidf_clf_result.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("已写:", p, flush=True)


if __name__ == "__main__":
    main()
