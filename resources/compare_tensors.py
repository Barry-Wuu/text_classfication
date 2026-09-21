# -*- coding: utf-8 -*-
"""Kaggle 云端：三条文本转张量策略的效果对比。

数据来源：从 GitHub 仓库直接拉 labeled.csv（已清洗 + 已编码 label）。
在同一 8:2 分层划分、同一 LinearSVC 分类头下，只替换"文本如何变张量"：

  A. BERT tokenizer : bert-base-chinese 逐字 → input_ids → [CLS] 768 维
  B. 词频自训词表    : jieba 分词 → 词频词表 → text_ids 均值池化 256 维
  C. TF-IDF 稀疏     : sklearn TfidfVectorizer

结果以 ===RESULT=== JSON 输出，并写入 /kaggle/working/result.json 与 run.log。
"""
import sys, os, json, time, traceback

_LOG = open('/kaggle/working/run.log', 'w', encoding='utf-8')
class _Tee:
    def __init__(self, *fs): self.fs = fs
    def write(self, s):
        for f in self.fs:
            try: f.write(s); f.flush()
            except Exception: pass
    def flush(self):
        for f in self.fs:
            try: f.flush()
            except Exception: pass
    def isatty(self): return False
    def fileno(self):
        for f in self.fs:
            try: return f.fileno()
            except Exception: pass
        return -1
    @property
    def encoding(self): return 'utf-8'
sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)
def _exchook(t, v, tb):
    txt = "".join(traceback.format_exception(t, v, tb))
    print("FATAL: " + txt, flush=True)
    try: open('/kaggle/working/FATAL.txt', 'w', encoding='utf-8').write(txt)
    except Exception: pass
sys.excepthook = _exchook

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
import scipy.sparse as sp

SEED = 42
MAX_LEN = 200
RAW_URL = "https://raw.githubusercontent.com/paixiaoxin66/text_classfication/Barry/labeled.csv"
MIRROR_URL = "https://ghfast.top/" + RAW_URL

def report(name, dim, Xtr, Xte, ytr, yte, extra=""):
    t0 = time.time()
    clf = LinearSVC(C=1.0)
    clf.fit(Xtr, ytr)
    fit_dt = time.time() - t0
    pred = clf.predict(Xte)
    f1 = f1_score(yte, pred, average="macro")
    acc = accuracy_score(yte, pred)
    row = dict(name=name, dim=int(dim), macro_f1=round(float(f1), 4),
               acc=round(float(acc), 4), fit_s=round(fit_dt, 2), extra=extra)
    print(f"  {name:14s} dim={dim:>6}  macroF1={f1:.4f}  acc={acc:.4f}  "
          f"训练={fit_dt:.1f}s  {extra}", flush=True)
    return row

def fetch_labeled():
    import requests
    dst = '/kaggle/working/labeled.csv'
    for url in (RAW_URL, MIRROR_URL):
        try:
            print("  下载:", url, flush=True)
            r = requests.get(url, timeout=120)
            print("    HTTP", r.status_code, "大小", len(r.content), flush=True)
            if r.status_code == 200 and len(r.content) > 1_000_000:
                open(dst, 'wb').write(r.content)
                return dst
        except Exception as e:
            print("    失败:", repr(e), flush=True)
    raise RuntimeError("labeled.csv 下载失败（raw 与镜像均不行）")

def main():
    print("=" * 60, flush=True)
    print("环境自检:", flush=True)
    import platform
    print("  python:", platform.python_version())
    import torch
    print("  torch:", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)

    csv = fetch_labeled()
    df = pd.read_csv(csv)
    print("  数据形状:", df.shape, "列:", list(df.columns), flush=True)
    y = df["label"].to_numpy(dtype="int64")
    texts = df["text"].astype(str).tolist()

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    ytr, yte = y[tr], y[te]
    print(f"训练 {len(tr)} / 测试 {len(te)}  类别数 {len(np.unique(y))}", flush=True)

    rows = []
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- A. BERT [CLS] 句向量 ----
    try:
        from transformers import BertTokenizerFast, BertModel
        tok = BertTokenizerFast.from_pretrained('bert-base-chinese')
        model = BertModel.from_pretrained('bert-base-chinese').to(dev).eval()
        t0 = time.time()
        ids_l, am_l = [], []
        for s in range(0, len(texts), 256):
            enc = tok(texts[s:s+256], max_length=MAX_LEN, truncation=True,
                      padding='max_length', return_tensors='pt')
            ids_l.append(enc['input_ids']); am_l.append(enc['attention_mask'])
        ids = torch.cat(ids_l).to(dev); am = torch.cat(am_l).to(dev)
        emb = []
        with torch.no_grad():
            for s in range(0, len(ids), 128):
                out = model(input_ids=ids[s:s+128], attention_mask=am[s:s+128])
                emb.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        Xb = np.vstack(emb)
        enc_dt = time.time() - t0
        print(f"  BERT 前向({dev}) {enc_dt:.1f}s → {Xb.shape}", flush=True)
        rows.append(report("A.BERT[CLS]", Xb.shape[1], Xb[tr], Xb[te], ytr, yte,
                           extra=f"(提取{enc_dt:.0f}s,{dev})"))
    except Exception as e:
        print("  [A] 失败:", repr(e), flush=True)

    # ---- B. 词频自训词表 + 均值池化 ----
    try:
        import jieba, re
        from collections import Counter
        _P = re.compile(r"^[\W_]+$", re.UNICODE)
        t0 = time.time()
        docs = [[w.strip() for w in jieba.lcut(t) if w.strip()] for t in texts]
        freq = Counter()
        for d in docs: freq.update(d)
        items = [(w, c) for w, c in freq.items() if c >= 2 and not _P.match(w)]
        items.sort(key=lambda x: (-x[1], x[0]))
        stoi = {w: i for i, w in enumerate(["<pad>", "<unk>"] + [w for w, _ in items])}
        tid = np.zeros((len(docs), MAX_LEN), dtype="int64")
        for i, d in enumerate(docs):
            seq = [stoi.get(w, 1) for w in d][:MAX_LEN]
            tid[i, :len(seq)] = seq
        cut_dt = time.time() - t0
        V = len(stoi)
        rng = np.random.default_rng(SEED)
        W = rng.standard_normal((V, 256)).astype("float32")
        mask = (tid != 0).astype("float32")[..., None]
        E = W[tid] * mask
        Xe = (E.sum(1) / mask.sum(1).clip(min=1))
        print(f"  词频: 词表={V} 分词{cut_dt:.1f}s → {Xe.shape}", flush=True)
        rows.append(report("B.词频Emb", Xe.shape[1], Xe[tr], Xe[te], ytr, yte,
                           extra=f"(词表{V})"))
    except Exception as e:
        print("  [B] 失败:", repr(e), flush=True)

    # ---- C. TF-IDF 稀疏 ----
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import jieba
        t0 = time.time()
        tfidf = TfidfVectorizer(tokenizer=lambda t: jieba.lcut(t), lowercase=False,
                                token_pattern=None, min_df=2)
        X = tfidf.fit_transform(texts)
        c_dt = time.time() - t0
        print(f"  TF-IDF: {X.shape} nnz={X.nnz} 构建{c_dt:.1f}s", flush=True)
        rows.append(report("C.TFIDF", X.shape[1], X[tr], X[te], ytr, yte,
                           extra=f"(nnz={X.nnz})"))
    except Exception as e:
        print("  [C] 失败:", repr(e), flush=True)

    best = max(rows, key=lambda r: r["macro_f1"]) if rows else None
    print("\n===RESULT_START===")
    print(json.dumps({"rows": rows, "best": best}, ensure_ascii=False))
    print("===RESULT_END===")
    json.dump({"rows": rows, "best": best},
              open('/kaggle/working/result.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
