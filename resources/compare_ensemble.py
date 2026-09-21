# -*- coding: utf-8 -*-
"""传统老机器学习"专家组"集成实验（无泄漏划分，与单机基线同协议）。

把 10 个不同类型的传统模型当成 10 位"专家"，用两种集成方式看能否顶个诸葛亮：
  - 硬投票 HardVoting：每位专家投一个类别，多数票胜出（平票用平均概率打破）
  - 软投票 SoftVoting：把各位专家的概率向量（SVM 用 decision_function 经 softmax 近似）
    求平均，取 argmax

报告 macro-F1 / accuracy，并给出理论上限 oracle（专家组中至少一位命中）与分歧率。
面板成员（与单机基线同超参，均吃同一份 TF-IDF 特征）：
  LinearSVC / SGD(hinge) / LogisticRegression / ComplementNB / MultinomialNB / kNN(cosine)
（不含树模型 RandomForest / ExtraTrees / XGBoost / LightGBM —— 用户指定仅测非树传统算法组合）
用法：python compare_ensemble.py
"""
import os, sys, json, time
import numpy as np
import pandas as pd
import jieba
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
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
    return np.concatenate([tr_ori, train_aug_idx]), te_ori, test_src


def softmax(x, axis=1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


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
    print(f"TF-IDF dim={Xtr.shape[1]} nnz={Xtr.nnz} 用时 {time.time()-t0:.1f}s", flush=True)
    t0 = time.time()
    Xtr_d = Xtr.toarray().astype("float32")
    Xte_d = Xte.toarray().astype("float32")
    print(f"转稠密 {Xtr_d.shape} 用时 {time.time()-t0:.1f}s", flush=True)

    # (name, clf, kind, has_native_proba)
    # 仅非树传统算法；树模型（RF/ET/XGB/LGB）按用户要求不参与本实验
    experts = [
        ("LinearSVC",           LinearSVC(C=1.0),                            "sparse", False),
        ("SGD(hinge)",          SGDClassifier(loss="hinge", random_state=SEED), "sparse", False),
        ("LogisticRegression",  LogisticRegression(max_iter=1000),          "sparse", True),
        ("ComplementNB",        ComplementNB(),                              "sparse", True),
        ("MultinomialNB",       MultinomialNB(),                             "sparse", True),
        ("kNN(cosine)",         KNeighborsClassifier(n_neighbors=15, metric="cosine", n_jobs=-1),        "dense", True),
    ]

    n = len(test)
    C = len(cats)
    labels_all = np.zeros((len(experts), n), dtype=int)
    prob_all = np.zeros((len(experts), n, C), dtype="float64")
    per_fit, per_inf = [], []
    panel_fit = 0.0

    for k, (name, clf, kind, has_proba) in enumerate(experts):
        Xa = Xtr if kind == "sparse" else Xtr_d
        Xb = Xte if kind == "sparse" else Xte_d
        try:
            t0 = time.time(); clf.fit(Xa, ytr); fit = time.time() - t0
            panel_fit += fit
            t0 = time.time(); pred = clf.predict(Xb); inf = time.time() - t0
            if has_proba:
                prob = clf.predict_proba(Xb).astype("float64")
            else:
                # SVM 类无 predict_proba：用 OvR decision_function 经 softmax 近似成概率
                prob = softmax(clf.decision_function(Xb).astype("float64"))
            f1 = f1_score(yte, pred, average="macro")
            acc = accuracy_score(yte, pred)
            print(f"  [专家{k+1:2d}] {name:18s} macroF1={f1:.4f} acc={acc:.4f} "
                  f"训练={fit:.1f}s 推理={inf:.1f}s", flush=True)
            labels_all[k] = pred
            prob_all[k] = prob
            per_fit.append(round(fit, 2)); per_inf.append(round(inf, 2))
        except Exception as e:
            print(f"  [专家{k+1:2d}] {name} 失败: {e!r}", flush=True)
            labels_all[k] = -1
            per_fit.append(None); per_inf.append(None)

    valid = [k for k in range(len(experts)) if per_fit[k] is not None]
    lab_v = labels_all[valid]
    prob_v = prob_all[valid]
    nv = len(valid)
    print(f"\n有效专家 {nv} 位：", [experts[k][0] for k in valid], flush=True)

    # ---- 硬投票（多数票；平票用平均概率打破）----
    t0 = time.time()
    votes = np.zeros((n, C), dtype=int)
    for lab in lab_v:
        votes[np.arange(n), lab] += 1
    mean_prob = prob_v.mean(axis=0)
    maxv = votes.max(axis=1, keepdims=True)
    tie = votes == maxv
    cand = np.where(tie, mean_prob, -1.0)
    hard_pred = cand.argmax(axis=1)
    hard_inf = time.time() - t0

    # ---- 软投票（平均概率）----
    t0 = time.time()
    soft_pred = prob_v.mean(axis=0).argmax(axis=1)
    soft_inf = time.time() - t0

    hard_f1 = f1_score(yte, hard_pred, average="macro")
    hard_acc = accuracy_score(yte, hard_pred)
    soft_f1 = f1_score(yte, soft_pred, average="macro")
    soft_acc = accuracy_score(yte, soft_pred)

    # ---- 诊断：理论上限 oracle（至少一位专家命中）+ 分歧率 ----
    oracle_correct = (lab_v == yte[None, :]).any(axis=0)
    oracle_acc = oracle_correct.mean()
    # oracle 的 macro-F1：把"任一专家正确"当作该样本命中，按类别统计
    oracle_pred = np.where(oracle_correct, yte, -1)  # 仅用于计数参考
    agree = (lab_v != lab_v[0:1, :]).any(axis=0)  # 任一专家与第一专家不同
    # 真正的分歧：并非所有有效专家一致
    consistent = np.all(lab_v == lab_v[0:1, :], axis=0)
    disagree_rate = 1.0 - consistent.mean()

    print("\n=== 集成结果 ===", flush=True)
    print(f"  硬投票 HardVoting   macroF1={hard_f1:.4f} acc={hard_acc:.4f} 投票耗时={hard_inf:.3f}s", flush=True)
    print(f"  软投票 SoftVoting   macroF1={soft_f1:.4f} acc={soft_acc:.4f} 投票耗时={soft_inf:.3f}s", flush=True)
    print(f"  理论上限 oracle(任一专家对) acc={oracle_acc:.4f} "
          f"（=若完美挑选专家可达的准确率上限）", flush=True)
    print(f"  专家分歧率（样本级）= {disagree_rate:.4f}（{int(disagree_rate*n)}/{n} 条至少两位专家意见不同）", flush=True)

    # ---- 合并入结果文件 ----
    p = os.path.join(ROOT, "resources", "tfidf_clf_result.json")
    d = json.load(open(p, encoding="utf-8"))
    rows = d.get("rows", [])
    new_rows = [
        dict(name="Ensemble:HardVoting", macro_f1=round(float(hard_f1), 4),
             acc=round(float(hard_acc), 4), fit_s=round(panel_fit, 2),
             infer_s=round(hard_inf, 4), extra=f"{nv}专家,硬投票,无泄漏"),
        dict(name="Ensemble:SoftVoting", macro_f1=round(float(soft_f1), 4),
             acc=round(float(soft_acc), 4), fit_s=round(panel_fit, 2),
             infer_s=round(soft_inf, 4), extra=f"{nv}专家,平均概率,无泄漏"),
    ]
    names = {r["name"] for r in new_rows}
    merged = [r for r in rows if r["name"] not in names] + new_rows
    merged.sort(key=lambda r: -r["macro_f1"])
    d["rows"] = merged
    d["best"] = merged[0]
    json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("已写:", p, flush=True)

    # ---- 诊断明细单独存（供 README 叙述）----
    diag = dict(
        panel=[experts[k][0] for k in valid],
        per_model=dict(
            zip([experts[k][0] for k in valid],
                [dict(macro_f1=round(float(f1_score(yte, labels_all[k], average="macro")), 4),
                      acc=round(float(accuracy_score(yte, labels_all[k])), 4),
                      fit_s=per_fit[k], infer_s=per_inf[k]) for k in valid])),
        hard_voting=dict(macro_f1=round(float(hard_f1), 4), acc=round(float(hard_acc), 4)),
        soft_voting=dict(macro_f1=round(float(soft_f1), 4), acc=round(float(soft_acc), 4)),
        oracle_acc=round(float(oracle_acc), 4),
        disagree_rate=round(float(disagree_rate), 4),
        n_experts=nv, n_test=n,
    )
    json.dump(diag, open(os.path.join(ROOT, "resources", "ensemble_diag.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("已写诊断:", os.path.join(ROOT, "resources", "ensemble_diag.json"), flush=True)


if __name__ == "__main__":
    main()
