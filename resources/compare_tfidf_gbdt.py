# -*- coding: utf-8 -*-
"""TF-IDF + 梯度提升树对比（XGBoost / LightGBM，本机 CPU，无泄漏划分）。

沿用 compare_tfidf_clf.py 的评估协议与同一份 TF-IDF 特征，
补测梯度提升树（GBDT）系模型：XGBoost / LightGBM。

要点：
  - 梯度提升树可直接吃 scipy 稀疏矩阵，无需像 RF/ET 那样转稠密（省内存）
  - 多分类用 softprob + mlogloss
用法：python compare_tfidf_gbdt.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import jieba
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

    t0 = time.time()
    tfidf = TfidfVectorizer(tokenizer=cut, lowercase=False, token_pattern=None, min_df=2)
    Xtr = tfidf.fit_transform([texts[i] for i in train])
    Xte = tfidf.transform([texts[i] for i in test])
    print(f"TF-IDF 特征：dim={Xtr.shape[1]} nnz={Xtr.nnz} 用时 {time.time()-t0:.1f}s", flush=True)

    rows = []

    # ---- XGBoost ----
    try:
        import xgboost as xgb
        t0 = time.time()
        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.15,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=len(cats),
            tree_method="hist", n_jobs=-1, random_state=SEED,
            eval_metric="mlogloss")
        clf.fit(Xtr, ytr)
        fit = time.time() - t0
        t0 = time.time(); pred = clf.predict(Xte); inf = time.time() - t0
        f1 = f1_score(yte, pred, average="macro"); acc = accuracy_score(yte, pred)
        print(f"  XGBoost   macroF1={f1:.4f} acc={acc:.4f} 训练={fit:.1f}s 推理={inf:.1f}s", flush=True)
        rows.append(dict(name="XGBoost", macro_f1=round(float(f1), 4),
                         acc=round(float(acc), 4), fit_s=round(fit, 2),
                         infer_s=round(inf, 2), extra="hist,400树"))
    except Exception as e:
        print("  [XGBoost] 失败:", repr(e), flush=True)

    # ---- LightGBM ----
    try:
        import lightgbm as lgb
        t0 = time.time()
        clf = lgb.LGBMClassifier(
            n_estimators=400, num_leaves=63, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="multiclass", num_class=len(cats),
            n_jobs=-1, random_state=SEED, verbose=-1)
        clf.fit(Xtr, ytr)
        fit = time.time() - t0
        t0 = time.time(); pred = clf.predict(Xte); inf = time.time() - t0
        f1 = f1_score(yte, pred, average="macro"); acc = accuracy_score(yte, pred)
        print(f"  LightGBM  macroF1={f1:.4f} acc={acc:.4f} 训练={fit:.1f}s 推理={inf:.1f}s", flush=True)
        rows.append(dict(name="LightGBM", macro_f1=round(float(f1), 4),
                         acc=round(float(acc), 4), fit_s=round(fit, 2),
                         infer_s=round(inf, 2), extra="leaf63,400树"))
    except Exception as e:
        print("  [LightGBM] 失败:", repr(e), flush=True)

    rows.sort(key=lambda r: -r["macro_f1"])
    print("\n=== 排名 ===")
    for r in rows:
        print(f"  {r['name']:12s} F1={r['macro_f1']:.4f} acc={r['acc']:.4f}")

    # 追加到已有结果文件
    p = os.path.join(ROOT, "resources", "tfidf_clf_result.json")
    prev = {}
    if os.path.exists(p):
        try: prev = json.load(open(p, encoding="utf-8"))
        except Exception: prev = {}
    prev_rows = prev.get("rows", [])
    names = {r["name"] for r in rows}
    merged = [r for r in prev_rows if r["name"] not in names] + rows
    merged.sort(key=lambda r: -r["macro_f1"])
    prev["rows"] = merged
    prev["best"] = merged[0] if merged else None
    json.dump(prev, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("已写:", p, flush=True)


if __name__ == "__main__":
    main()
