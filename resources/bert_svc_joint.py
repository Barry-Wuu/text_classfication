# -*- coding: utf-8 -*-
"""bert_svc_joint.py — 逐样本联合分布：LinearSVC 判对 / BERT 判对 的交集。

给 5.10 的韦恩图提供精确数字（不用估算或区间）。划分与
compare_tfidf_clf.py / compare_bert_finetune.py 完全一致（同一份
GroupShuffleSplit，random_state=42）；BERT 用 Release 权重
bert_epoch59.pt 的 state_dict 在本地 CPU 推理，全测试集 1,378 条。

输出：resources/bert_svc_joint.json
"""
import os
import json

import numpy as np
import pandas as pd
import jieba
import torch
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from transformers import BertTokenizerFast, BertForSequenceClassification

jieba.setLogLevel(20)
SEED = 42
MAX_LEN = 128
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = r"D:\models\bert-base-chinese"
WEIGHT = r"D:\B++\WorkBuddy\bert_epoch59.pt"

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
yte = y[te]
print(f"train {len(train)} / test {len(te)}", flush=True)

# ---- LinearSVC（词级 TF-IDF，与 5.5 完全同参） ----
tf = TfidfVectorizer(tokenizer=jieba.lcut, lowercase=False,
                     token_pattern=None, min_df=2)
Xtr = tf.fit_transform([texts[i] for i in train])
Xte = tf.transform([texts[i] for i in te])
svc = LinearSVC(C=1.0)
svc.fit(Xtr, y[train])
svc_pred = svc.predict(Xte)
print("LinearSVC acc = %.4f" % accuracy_score(yte, svc_pred), flush=True)

# ---- BERT CPU 推理 ----
tok = BertTokenizerFast.from_pretrained(BASE)
model = BertForSequenceClassification.from_pretrained(BASE, num_labels=len(cats))
sd = torch.load(WEIGHT, map_location="cpu")
model.load_state_dict(sd["state_dict"])
model.eval()
print("weights loaded", flush=True)

bert_pred = []
with torch.no_grad():
    for s in range(0, len(te), 64):
        batch_texts = [texts[i] for i in te[s:s + 64]]
        enc = tok(batch_texts, max_length=MAX_LEN, truncation=True,
                  padding=True, return_tensors="pt")
        logits = model(input_ids=enc["input_ids"],
                       attention_mask=enc["attention_mask"]).logits
        bert_pred.extend(logits.argmax(-1).tolist())
        if s % 320 == 0:
            print(f"  {s + len(batch_texts)}/{len(te)}", flush=True)
bert_pred = np.array(bert_pred)
print("BERT acc = %.4f" % accuracy_score(yte, bert_pred), flush=True)

# ---- 联合分布 ----
svc_ok = svc_pred == yte
bert_ok = bert_pred == yte
joint = {}
for k, c in enumerate(cats):
    idx = yte == k
    n = int(idx.sum())
    joint[c] = {
        "n": n,
        "both": int((svc_ok & bert_ok)[idx].sum()),
        "svc_only": int((svc_ok & ~bert_ok)[idx].sum()),
        "bert_only": int((~svc_ok & bert_ok)[idx].sum()),
        "neither": int((~svc_ok & ~bert_ok)[idx].sum()),
    }
joint["_overall"] = {
    "n": int(len(te)),
    "both": int((svc_ok & bert_ok).sum()),
    "svc_only": int((svc_ok & ~bert_ok).sum()),
    "bert_only": int((~svc_ok & bert_ok).sum()),
    "neither": int((~svc_ok & ~bert_ok).sum()),
    "svc_acc": float(svc_ok.mean()),
    "bert_acc": float(bert_ok.mean()),
}
out = os.path.join(ROOT, "resources", "bert_svc_joint.json")
json.dump(joint, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved", out)
for c in ("电商平台", "影音娱乐", "家居日用", "共享出行", "_overall"):
    j = joint[c]
    print(f"{c:8s} n={j['n']:4d}  都对={j['both']:4d}  仅SVC={j['svc_only']:3d}"
          f"  仅BERT={j['bert_only']:3d}  都错={j['neither']:3d}")
