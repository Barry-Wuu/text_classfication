# -*- coding: utf-8 -*-
"""张量方案 B：jieba 分词 + 词频自训词表。

流程：
  1) jieba 分词（全量）
  2) 统计词频，按词频降序构建词表
     - 特殊符号：<pad>=0, <unk>=1
     - 过滤：去纯空白 / 纯标点；min_freq 默认 2
  3) 词 → id 映射，截断/补齐到 200，得整数序列张量
  4) 另存 TF-IDF 稀疏矩阵（sklearn，对照用）

产物：
  resources/vocab.txt         一行一词，行号 = id（0=<pad>,1=<unk>）
  resources/tensors_tfidf.npz text_ids [N,200] int32, labels [N], lengths [N]
  resources/tfidf_matrix.npz  scipy CSR（data/indices/indptr/shape）
  resources/tfidf_vectorizer.pkl
"""
import os
import re
import time
import pickle
import numpy as np
import pandas as pd
import jieba
from collections import Counter
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = r"D:\text_classfication"
RES = os.path.join(ROOT, "resources")
MAX_LEN = 200
MIN_FREQ = 2
PAD, UNK = 0, 1

_PUNCT_ONLY = re.compile(r"^[\W_]+$", re.UNICODE)

def cut(text: str):
    return [w.strip() for w in jieba.lcut(str(text)) if w.strip()]

def main():
    df = pd.read_csv(os.path.join(ROOT, "labeled.csv"))
    texts = df["text"].astype(str).tolist()
    labels = df["label"].to_numpy(dtype="int64")

    t0 = time.time()
    docs = [cut(t) for t in texts]
    cut_dt = time.time() - t0

    # 词频
    freq = Counter()
    for doc in docs:
        freq.update(doc)

    # 构建词表：过滤纯标点/纯空白，min_freq 过滤，按 (频次降序, 词) 固定顺序
    items = [(w, c) for w, c in freq.items()
             if c >= MIN_FREQ and not _PUNCT_ONLY.match(w)]
    items.sort(key=lambda x: (-x[1], x[0]))

    itos = ["<pad>", "<unk>"] + [w for w, _ in items]
    stoi = {w: i for i, w in enumerate(itos)}

    with open(os.path.join(RES, "vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(itos) + "\n")

    # 序列编码
    n = len(docs)
    ids = np.zeros((n, MAX_LEN), dtype="int32")   # 默认 0 = <pad>
    lengths = np.zeros(n, dtype="int32")
    for i, doc in enumerate(docs):
        seq = [stoi.get(w, UNK) for w in doc][:MAX_LEN]
        lengths[i] = len(seq)
        ids[i, :len(seq)] = seq

    np.savez_compressed(os.path.join(RES, "tensors_tfidf.npz"),
                        text_ids=ids, labels=labels, lengths=lengths)

    # TF-IDF 稀疏矩阵（整语料词表，对照）
    tfidf = TfidfVectorizer(tokenizer=cut, lowercase=False,
                            token_pattern=None, min_df=2)
    X = tfidf.fit_transform(texts)
    sparse.save_npz(os.path.join(RES, "tfidf_matrix.npz"), X.tocsr())
    with open(os.path.join(RES, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)

    vocab_used = int((ids != PAD).sum())
    print(f"[OK] jieba 分词耗时={cut_dt:.1f}s  词表大小={len(itos)}"
          f"（min_freq>={MIN_FREQ}）")
    print(f"     text_ids shape={ids.shape}  非pad元素={vocab_used}")
    print(f"     length P50={int(np.percentile(lengths,50))} "
          f"P95={int(np.percentile(lengths,95))} P99={int(np.percentile(lengths,99))} "
          f"max={int(lengths.max())}  截断={int((lengths>MAX_LEN).sum())}")
    print(f"     TF-IDF 矩阵 shape={X.shape} nnz={X.nnz}"
          f"  体积={os.path.getsize(os.path.join(RES,'tfidf_matrix.npz'))/1e6:.1f} MB")
    print(f"     tensors_tfidf.npz 体积="
          f"{os.path.getsize(os.path.join(RES,'tensors_tfidf.npz'))/1e6:.1f} MB")
    # 前 20 高频词
    print("     Top20:", " ".join(f"{w}({c})" for w, c in items[:20]))

if __name__ == "__main__":
    main()
