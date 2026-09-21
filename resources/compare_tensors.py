# -*- coding: utf-8 -*-
"""对比两条文本转张量策略的效果与成本（子集快速版）。

目的：选出**更适合本数据集的张量策略**，而非比拼分类器。
在同一子集（默认 8000 条，stratify 抽样）上，用同一线性分类头 LinearSVC，
只替换"文本如何变张量"这一步：

  A. BERT tokenizer : bert-base-chinese 逐字 → input_ids，取 [CLS] 768 维
  B. 词频自训词表    : jieba 分词 → 词表 → text_ids 均值池化 256 维
  C. TF-IDF 稀疏     : sklearn TfidfVectorizer（传统基线）

指标：macro-F1（主）、张量编码耗时、特征维度、张量落盘体积。
"""
import os
import time
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score

ROOT = r"D:\text_classfication"
RES = os.path.join(ROOT, "resources")
SEED = 42
N_SUB = 8000          # 子集规模（含 train+test）

def report(name, dim, Xtr, Xte, ytr, yte, extra=""):
    t0 = time.time()
    clf = LinearSVC(C=1.0)
    clf.fit(Xtr, ytr)
    fit_dt = time.time() - t0
    pred = clf.predict(Xte)
    f1 = f1_score(yte, pred, average="macro")
    acc = accuracy_score(yte, pred)
    print(f"  {name:14s} dim={dim:>5}  macroF1={f1:.4f}  acc={acc:.4f}  "
          f"训练={fit_dt:.1f}s  {extra}", flush=True)
    return f1

def main():
    y_all = np.load(os.path.join(RES, "tensors_bert.npz"))["labels"]
    idx_all = np.arange(len(y_all))
    # 先抽子集（分层），再在子集内 8:2 划分
    sub, _ = train_test_split(idx_all, train_size=N_SUB, random_state=SEED,
                              stratify=y_all)
    tr, te = train_test_split(sub, test_size=0.2, random_state=SEED,
                              stratify=y_all[sub])
    ytr, yte = y_all[tr], y_all[te]
    print(f"子集 {N_SUB} → 训练 {len(tr)} / 测试 {len(te)}  类别 {len(np.unique(y_all))}", flush=True)

    # ---- A. BERT [CLS] 句向量 ----
    import torch
    from transformers import BertModel
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    z = np.load(os.path.join(RES, "tensors_bert.npz"))
    ids = torch.from_numpy(z["input_ids"])[sub].long()
    am = torch.from_numpy(z["attention_mask"])[sub].long()
    model = BertModel.from_pretrained(r"D:\models\bert-base-chinese").eval()
    emb = []
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, len(ids), 128):
            out = model(input_ids=ids[s:s+128], attention_mask=am[s:s+128])
            emb.append(out.last_hidden_state[:, 0, :].numpy())
    Xbert = np.vstack(emb)
    bert_dt = time.time() - t0
    print(f"  BERT 前向 {bert_dt:.1f}s → 句向量 {Xbert.shape}", flush=True)
    # 把子集下标映射回本地下标
    pos = {g: i for i, g in enumerate(sub)}
    tri = np.array([pos[g] for g in tr]); tei = np.array([pos[g] for g in te])
    f1A = report("A.BERT[CLS]", Xbert.shape[1], Xbert[tri], Xbert[tei], ytr, yte,
                 extra=f"(提取 {bert_dt:.0f}s)")

    # ---- B. 词频 EmbeddingBag（随机嵌入，均值池化）----
    zz = np.load(os.path.join(RES, "tensors_tfidf.npz"))
    tid = torch.from_numpy(zz["text_ids"])[sub].long()
    vocab_sz = int(torch.from_numpy(zz["text_ids"]).max()) + 1
    rng = torch.Generator().manual_seed(SEED)
    W = torch.randn(vocab_sz, 256, generator=rng)
    with torch.no_grad():
        E = W[tid]
        mask = (tid != 0).unsqueeze(-1).float()
        Xemb = ((E * mask).sum(1) / mask.sum(1).clamp(min=1)).numpy()
    f1B = report("B.词频Emb", Xemb.shape[1], Xemb[tri], Xemb[tei], ytr, yte,
                 extra="(随机嵌入)")

    # ---- C. TF-IDF 稀疏（子集切片）----
    X = sp.load_npz(os.path.join(RES, "tfidf_matrix.npz"))[sub]
    f1C = report("C.TFIDF", X.shape[1], X[tri], X[tei], ytr, yte,
                 extra=f"(nnz={X.nnz})")

    best = max([("A.BERT[CLS]", f1A), ("B.词频Emb", f1B), ("C.TFIDF", f1C)],
               key=lambda kv: kv[1])
    print(f"\n[结论] 子集({N_SUB})上最优张量策略: {best[0]}  macroF1={best[1]:.4f}")

if __name__ == "__main__":
    main()
