# -*- coding: utf-8 -*-
"""严格无泄漏的对比：按"源原始样本 id"分组划分 train/test。

labeled.csv 没有源信息，这里回到 train_augmented.csv：
  - 原始样本 id = 纯数字（如 17401683207）
  - 增强样本 id = 源id + '#' + 变体号（如 17401489078#4）
  - 源 id = id.split('#')[0]

用 GroupShuffleSplit 按源 id 分组，保证同一源的所有样本（原始+全部变体）
要么全在 train，要么全在 test —— 彻底杜绝"同源变体跨集"的隐蔽泄漏。

对比：
  1) 普通随机划分（同分布，可能有同源跨集）
  2) 分组划分（按源 id）
  两种划分下各跑 TF-IDF + LinearSVC，看分差有多大。
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = r"D:\text_classfication"
SEED = 42

def run(name, tr, te, texts, y, jieba_cut):
    tfidf = TfidfVectorizer(tokenizer=jieba_cut, lowercase=False,
                            token_pattern=None, min_df=2)
    Xtr = tfidf.fit_transform([texts[i] for i in tr])
    Xte = tfidf.transform([texts[i] for i in te])
    clf = LinearSVC(C=1.0)
    t0 = time.time()
    clf.fit(Xtr, y[tr])
    dt = time.time() - t0
    pred = clf.predict(Xte)
    f1 = f1_score(y[te], pred, average="macro")
    acc = accuracy_score(y[te], pred)
    print(f"  {name:22s} train={len(tr):>5} test={len(te):>5}  "
          f"macroF1={f1:.4f} acc={acc:.4f} 训练={dt:.1f}s", flush=True)
    return f1, acc

def main():
    import jieba
    cut = lambda t: jieba.lcut(t)

    df = pd.read_csv(os.path.join(ROOT, "train_augmented.csv"))
    texts = df["text"].astype(str).tolist()
    # 标签编码（字典序，与 build_labels 一致）
    cats = sorted(df["category"].unique())
    m = {c: i for i, c in enumerate(cats)}
    y = df["category"].map(m).to_numpy()
    src = df["id"].astype(str).str.split("#").str[0].to_numpy()   # 源 id
    print("样本:", len(df), " 唯一源 id:", len(np.unique(src)), flush=True)

    print("\n[1] 普通随机划分（可能有同源变体跨 train/test）", flush=True)
    idx = np.arange(len(df))
    tr1, te1 = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    run("随机划分", tr1, te1, texts, y, cut)

    print("\n[2] 按源 id 分组划分（同源族不跨集）", flush=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr2, te2 = next(gss.split(idx, y, groups=src))
    run("分组划分", tr2, te2, texts, y, cut)

    # 量化泄漏规模：随机划分下有多少测试样本，其同源族出现在训练集
    tr_src = set(src[tr1])
    leak = sum(1 for i in te1 if src[i] in tr_src)
    print(f"\n随机划分下，测试集中 {leak}/{len(te1)} ({leak/len(te1)*100:.1f}%) "
          f"的样本，其同源族也出现在训练集 → 这就是泄漏规模", flush=True)

if __name__ == "__main__":
    main()
