# -*- coding: utf-8 -*-
"""TF-IDF/fastText 对比：fastText 监督分类（本机 CPU，无泄漏划分）。

沿用 compare_tfidf_clf.py 的评估协议与同一划分，补测 fastText：
  - fastText 是"词袋 + 词向量 + 线性分类"一体的高效基线，天然吃原始文本
  - 中文需先分词（jieba），否则 fastText 按整句/字符切分效果差
  - 训练用 `train_supervised`，predict 走 `model.predict`

与其它模型同口径输出 macro-F1 / 准确率 / 训练耗时，结果 merge 进
resources/tfidf_clf_result.json。
用法：python compare_fasttext.py
"""
import os, sys, json, time, tempfile
import numpy as np
import pandas as pd
import jieba
from sklearn.model_selection import GroupShuffleSplit
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


def write_ft_file(path, texts, labels):
    """fastText 格式：__label__<id> <空格分词后的文本>"""
    with open(path, "w", encoding="utf-8") as f:
        for t, y in zip(texts, labels):
            seg = " ".join(w for w in cut(str(t)) if w.strip())
            f.write(f"__label__{y} {seg}\n")


def main():
    import fasttext
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

    tmp = tempfile.mkdtemp(prefix="ft_")
    tr_f = os.path.join(tmp, "train.txt")
    te_f = os.path.join(tmp, "test.txt")
    t0 = time.time()
    write_ft_file(tr_f, [texts[i] for i in train], ytr)
    write_ft_file(te_f, [texts[i] for i in test], yte)
    print(f"写出 fastText 数据用时 {time.time()-t0:.1f}s", flush=True)

    rows = []
    configs = [
        ("fastText", dict(dim=100, epoch=25, lr=0.5, word_ngrams=2, minn=2, maxn=5, loss="softmax", thread=8)),
    ]
    for name, kw in configs:
        try:
            t0 = time.time()
            model = fasttext.train_supervised(input=tr_f, **kw)
            fit = time.time() - t0
            # 用语料自带 test 文件评估
            t0 = time.time()
            N, prec, rec = model.test(te_f, k=1)
            inf = time.time() - t0
            # 用 sklearn 口径复算 macro-F1
            lines = open(te_f, encoding="utf-8").read().splitlines()
            gold = [int(l.split()[0].replace("__label__", "")) for l in lines]
            txts = [" ".join(l.split()[1:]) for l in lines]
            preds = [int(model.predict(t, k=1)[0][0].replace("__label__", "")) for t in txts]
            f1 = f1_score(gold, preds, average="macro")
            acc = accuracy_score(gold, preds)
            print(f"  {name}  macroF1={f1:.4f} acc={acc:.4f} "
                  f"训练={fit:.1f}s 推理={inf:.1f}s（ft自带 prec={prec:.4f}）", flush=True)
            rows.append(dict(name=name, macro_f1=round(float(f1), 4),
                             acc=round(float(acc), 4), fit_s=round(fit, 2),
                             infer_s=round(inf, 2), extra="dim100,ep25,ngram2-5"))
        except Exception as e:
            print(f"  {name} 失败:", repr(e), flush=True)

    rows.sort(key=lambda r: -r["macro_f1"])
    print("\n=== 排名 ===")
    for r in rows:
        print(f"  {r['name']:12s} F1={r['macro_f1']:.4f} acc={r['acc']:.4f}")

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
