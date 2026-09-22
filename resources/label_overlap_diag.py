# -*- coding: utf-8 -*-
"""标签质量诊断：哪些类是"天生难分"（类间语义重叠），哪些只是模型没学好。

五个判据（全部不依赖单个模型的偏好）：
  1) 集体错率：4 个归纳偏置不同的模型（词级 NB/SVM/SVM-hinge + 字符级 SVM）
     对同一测试样本 >=3 个判错的比例。集体错 => 词面上就难分。
     由它得 oracle 上界：1 - 集体错率 >= 任意单模型可达上限。
  2) 类中心相似度：训练集 TF-IDF 类中心两两余弦，找最相似的类对（无监督）。
  3) 无模型最近邻自洽率：测试样本最近的类中心是否为自己的类。
  4) 漂移去向：自洽失败的样本被吸去了哪个类。
  5) 标注噪声排查：原文完全相同 / 前 30 字相同但标签不同的条数。
     若接近 0，则重叠来自"标注定义边界"，不是脏数据。

划分与 compare_tfidf_clf.py 完全一致（GroupShuffleSplit, random_state=42）。
输出：resources/label_quality_diag.json
"""
import os
import json
from collections import Counter

import numpy as np
import pandas as pd
import jieba
from scipy import sparse
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score

jieba.setLogLevel(20)
SEED = 42
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOCUS = ["电商平台", "影音娱乐", "家居日用", "共享出行"]

df = pd.read_csv(os.path.join(ROOT, "train_augmented.csv"))
cats = sorted(df["category"].unique())
lab = {c: i for i, c in enumerate(cats)}
y = df["category"].map(lab).to_numpy()
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
Xtr_t = [texts[i] for i in train]
Xte_t = [texts[i] for i in te]
ytr, yte = y[train], y[te]
print("train %d / test %d" % (len(train), len(te)))

# ---------- 1) 四模型预测 ----------
preds = {}
tf = TfidfVectorizer(tokenizer=jieba.lcut, lowercase=False,
                     token_pattern=None, min_df=2)
Xw = tf.fit_transform(Xtr_t)
Xw_te = tf.transform(Xte_t)
for nm, clf in [("ComplementNB", ComplementNB()),
                ("SGD_hinge", SGDClassifier(loss="hinge", random_state=SEED)),
                ("LinearSVC", LinearSVC(C=1.0))]:
    clf.fit(Xw, ytr)
    preds[nm] = clf.predict(Xw_te)
    print("  %-12s acc=%.4f" % (nm, accuracy_score(yte, preds[nm])))
tfc = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=12)
Xc = tfc.fit_transform(Xtr_t)
Xc_te = tfc.transform(Xte_t)
clf_c = LinearSVC(C=1.0)
clf_c.fit(Xc, ytr)
preds["LinearSVC_char"] = clf_c.predict(Xc_te)
print("  %-12s acc=%.4f (char ngram, %d feats)"
      % ("LinearSVC_char", accuracy_score(yte, preds["LinearSVC_char"]), Xc.shape[1]))

names = list(preds)
wrong = {nm: (preds[nm] != yte) for nm in names}
w4 = np.sum([wrong[n] for n in names], axis=0)

hard4, oracle = {}, {}
for k, c in enumerate(cats):
    idx = np.where(yte == k)[0]
    if len(idx) == 0:
        continue
    hard4[c] = float((w4[idx] >= 3).mean())
    oracle[c] = 1.0 - hard4[c]
acc4 = {nm: float(accuracy_score(yte, preds[nm])) for nm in names}

# ---------- 2) 类中心相似度（全 18x18） ----------
def centroids(X, yy):
    Z = np.zeros((len(cats), X.shape[1]))
    for k in range(len(cats)):
        idx = np.where(yy == k)[0]
        Z[k] = np.asarray(X[idx].mean(axis=0)).ravel()
    return Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-9)

C = centroids(Xw, ytr)
S = C @ C.T
np.fill_diagonal(S, -1)
iu = np.triu_indices(len(cats), 1)
pairs = sorted(zip(S[iu], [cats[a] for a in iu[0]], [cats[b] for b in iu[1]]),
               reverse=True)
sim_off = S[iu]
top_pairs = [[a, b, float(s)] for s, a, b in pairs[:10]]

# ---------- 3) 无模型最近邻自洽率 + 4) 漂移去向 ----------
Zt = tf.transform(Xte_t)
# 稀疏矩阵逐行归一不能用 np.linalg.norm（会把它当标量乘法，且本机内存吃紧），
# 用 sklearn 的稀疏感知 normalize
Zt = normalize(Zt, norm="l2", copy=False)
nn = np.asarray(Zt @ C.T).argmax(axis=1)
self_rate, drift = {}, {}
for k, c in enumerate(cats):
    idx = np.where(yte == k)[0]
    if len(idx) == 0:
        continue
    self_rate[c] = float((nn[idx] == k).mean())
    miss = idx[nn[idx] != k]
    drift[c] = dict(Counter(cats[nn[i]] for i in miss).most_common(5))

# ---------- 5) 标注噪声排查 ----------
g_all = df[df.is_aug == 0].groupby("text")["category"].nunique()
same_text_cross = int((g_all > 1).sum())
pref = df[df.is_aug == 0].assign(p=df["text"].astype(str).str[:30])
gp = pref.groupby("p")["category"].nunique()
pref_cross = int((gp > 1).sum())

# ---------- 焦点类补充：LinearSVC / BERT 的主要误判对 ----------
cm_json = json.load(open(os.path.join(ROOT, "resources", "confusion_matrices.json"),
                         encoding="utf-8"))
bert_json = json.load(open(os.path.join(ROOT, "resources", "bert_confusion.json"),
                           encoding="utf-8"))
focus_conf = {}
cm_lin = cm_json["models"]["LinearSVC"]["matrix"]
bert_lin = bert_json["matrix"]
for c in FOCUS:
    i = cats.index(c)
    row = {}
    for nm, mat in [("LinearSVC", cm_lin), ("BERT", bert_lin)]:
        M = np.array(mat)
        off = M[i].copy()
        off[i] = 0
        rec = float(M[i, i] / M[i].sum())
        top = [(cats[a], int(off[a])) for a in np.argsort(-off)[:3] if off[a] > 0]
        row[nm] = {"recall": rec, "top_wrong_to": top}
    focus_conf[c] = row

out = {
    "protocol": "GroupShuffleSplit(test_size=0.2, random_state=42), 无泄漏",
    "n_test": int(len(te)),
    "model_acc": acc4,
    "collective_wrong_rate": hard4,
    "oracle_bound_1_minus_hard4": oracle,
    "centroid_sim_top_pairs": top_pairs,
    "centroid_sim_mean": float(sim_off.mean()),
    "self_rate": self_rate,
    "drift_top": drift,
    "same_text_cross_label": same_text_cross,
    "same_prefix30_cross_label": pref_cross,
    "focus_confusion": focus_conf,
}
with open(os.path.join(ROOT, "resources", "label_quality_diag.json"), "w",
          encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

print()
print("集体错率(>=3/4) Top6:")
for c, v in sorted(hard4.items(), key=lambda x: -x[1])[:6]:
    print("   %-8s %.1f%%   oracle上界 %.1f%%" % (c, v * 100, oracle[c] * 100))
print("最相似类对 Top5:")
for a, b, s in top_pairs[:5]:
    print("   %s - %s  %.3f" % (a, b, s))
print("自洽率最低 6:", {c: round(v, 3) for c, v in
                      sorted(self_rate.items(), key=lambda x: x[1])[:6]})
for c in FOCUS:
    print("漂移", c, "->", drift[c])
print("同文跨类 %d / 前30字同跨类 %d" % (same_text_cross, pref_cross))
