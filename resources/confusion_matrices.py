# -*- coding: utf-8 -*-
"""四个帕累托最优模型在无泄漏测试集上的 18 阶混淆矩阵（计数）。

三个本地模型（ComplementNB / SGD(hinge) / LinearSVC）在本机复算，
严格沿用 compare_tfidf_clf.py 的超参与划分协议（测试=1378 全原始，源 id 不相交）；
BERT 由 Kaggle 内核算出后合并进同一份 JSON。

输出：resources/confusion_matrices.json
"""
import os, json, time
import numpy as np
import pandas as pd
import jieba
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "resources")
CSV = os.path.join(ROOT, "train_augmented.csv")
SEED = 42

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
    return np.concatenate([tr_ori, train_aug_idx]), te_ori, test_src


def main():
    df = pd.read_csv(CSV)
    cats = sorted(df["category"].unique())
    m = {c: i for i, c in enumerate(cats)}
    y = df["category"].map(m).to_numpy()
    texts = df["text"].astype(str).tolist()
    train, test, test_src = build_split(df)
    ytr, yte = y[train], y[test]
    print("类别 %d | 训练 %d | 测试 %d | 测试源 %d"
          % (len(cats), len(train), len(test), len(test_src)), flush=True)

    t0 = time.time()
    tfidf = TfidfVectorizer(tokenizer=cut, lowercase=False, token_pattern=None, min_df=2)
    Xtr = tfidf.fit_transform([texts[i] for i in train])
    Xte = tfidf.transform([texts[i] for i in test])
    print("TF-IDF dim=%d 用时 %.1fs" % (Xtr.shape[1], time.time() - t0), flush=True)

    models = [
        ("ComplementNB", ComplementNB()),
        ("SGD(hinge)",   SGDClassifier(loss="hinge", random_state=SEED)),
        ("LinearSVC",    LinearSVC(C=1.0)),
    ]

    out = {"labels": cats,
           "split": {"train": int(len(train)), "test": int(len(test)),
                     "test_sources": int(len(test_src)),
                     "test_dist": [int((yte == k).sum()) for k in range(len(cats))]},
           "models": {}}

    for name, clf in models:
        t0 = time.time()
        clf.fit(Xtr, ytr)
        fit = time.time() - t0
        pred = clf.predict(Xte)
        cm = confusion_matrix(yte, pred, labels=list(range(len(cats))))
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average="macro")
        print("  %-14s acc=%.4f macroF1=%.4f 用时=%.2fs" % (name, acc, f1, fit), flush=True)
        out["models"][name] = {
            "matrix": cm.tolist(), "acc": round(float(acc), 4),
            "macro_f1": round(float(f1), 4), "fit_s": round(fit, 2),
            "row_sum": cm.sum(axis=1).tolist(),
        }

    p = os.path.join(RES, "confusion_matrices.json")
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("已写:", p, flush=True)


if __name__ == "__main__":
    main()
