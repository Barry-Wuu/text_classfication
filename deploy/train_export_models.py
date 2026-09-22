# -*- coding: utf-8 -*-
"""训练并导出部署用的两个非 BERT 模型（与 5.5 报告完全同协议）。

- 同一无泄漏分组划分（GroupShuffleSplit, random_state=42）训练 + 评测；
- 同一 TF-IDF 配置（jieba 分词、min_df=2）；
- 产出 joblib 模型与标签表，附带回测指标（应与 5.5 表逐位一致）。

用法：python deploy/train_export_models.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.naive_bayes import ComplementNB

from jieba_cut import cut   # 模块级函数, 便于 joblib 按引用序列化
SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(OUT, exist_ok=True)

df = pd.read_csv(os.path.join(ROOT, "train_augmented.csv"))
cats = sorted(df["category"].unique())
y = df["category"].map({c: i for i, c in enumerate(cats)}).to_numpy()
texts = df["text"].astype(str).tolist()
src = df["id"].astype(str).str.split("#").str[0]
is_aug = df["is_aug"].to_numpy()

ori = np.where(is_aug == 0)[0]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
tr, te = next(gss.split(np.arange(len(df))[ori], groups=src.to_numpy()[ori]))
tr, te = ori[tr], ori[te]
test_src = set(src.to_numpy()[te])
aug_keep = np.array([i for i in np.where(is_aug == 1)[0]
                     if src.to_numpy()[i] not in test_src])
train = np.concatenate([tr, aug_keep])
print("train %d / test %d" % (len(train), len(te)))

vec = TfidfVectorizer(tokenizer=cut, lowercase=False,
                      token_pattern=None, min_df=2)
Xtr = vec.fit_transform([texts[i] for i in train])
Xte = vec.transform([texts[i] for i in te])
ytr, yte = y[train], y[te]

metrics = {}
for key, clf in (("complementnb", ComplementNB()),
                 ("sgd_hinge", SGDClassifier(loss="hinge", random_state=SEED))):
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    f1 = float(f1_score(yte, pred, average="macro"))
    joblib.dump(clf, os.path.join(OUT, key + ".joblib"), compress=3)
    sz = os.path.getsize(os.path.join(OUT, key + ".joblib")) / 1048576
    metrics[key] = {"acc": round(acc, 4), "macro_f1": round(f1, 4),
                    "size_mb": round(sz, 2)}
    print("%-14s acc=%.4f f1=%.4f size=%.2fMB" % (key, acc, f1, sz), flush=True)

joblib.dump(vec, os.path.join(OUT, "tfidf_vectorizer.joblib"), compress=3)
json.dump(cats, open(os.path.join(OUT, "labels.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
json.dump({"protocol": "GroupShuffleSplit(test_size=0.2, random_state=42)",
           "train": int(len(train)), "test": int(len(te)),
           "tfidf": {"tokenizer": "jieba_cut.cut", "min_df": 2,
                     "n_features": int(Xtr.shape[1])},
           "metrics": metrics},
          open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print("导出完成 ->", OUT)
for f in sorted(os.listdir(OUT)):
    print("   %8.2f MB  %s" % (os.path.getsize(os.path.join(OUT, f)) / 1048576, f))
