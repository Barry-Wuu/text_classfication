# -*- coding: utf-8 -*-
"""无泄漏重比：分组划分（按源 id）+ 训练含增强 / 测试只用原始样本。

真实场景模拟：
  - 测试集 = 原始样本中，按"源 id"分层切出的 20%（这些源的所有增强变体
    都不进训练集 —— 彻底杜绝同源泄漏）
  - 训练集 = 剩余 80% 源的原始样本 + 所有非测试源的增强样本

对比三条张量策略（同一划分、同一分类头）：
  A. BERT 微调（bert-base-chinese，端到端）
  B. TF-IDF + LinearSVC
  C. 词频随机嵌入 + LinearSVC

数据：从 GitHub 拉 train_augmented.csv（含 id/is_aug/variant_idx/category）。
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

SEED = 42
MAX_LEN = 200
RAW_URL = "https://raw.githubusercontent.com/Barry-Wuu/text_classfication/main/train_augmented.csv"
MIRROR_URL = "https://ghfast.top/" + RAW_URL

def fetch():
    import requests
    dst = '/kaggle/working/train_augmented.csv'
    for url in (RAW_URL, MIRROR_URL):
        try:
            print("  下载:", url, flush=True)
            r = requests.get(url, timeout=180)
            print("    HTTP", r.status_code, "大小", len(r.content), flush=True)
            if r.status_code == 200 and len(r.content) > 1_000_000:
                open(dst, 'wb').write(r.content); return dst
        except Exception as e:
            print("    失败:", repr(e), flush=True)
    raise RuntimeError("数据下载失败")

def build_split(df):
    """分组划分：测试只含原始样本，其源族不进训练。"""
    src = df["id"].astype(str).str.split("#").str[0]
    is_aug = df["is_aug"].to_numpy()
    # 只对"原始样本"按源分组切 20%
    ori_idx = np.where(is_aug == 0)[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    ori_all = np.arange(len(df))
    tr_ori, te_ori = next(gss.split(ori_all[ori_idx], groups=src.to_numpy()[ori_idx]))
    tr_ori = ori_idx[tr_ori]; te_ori = ori_idx[te_ori]
    test_src = set(src.to_numpy()[te_ori])
    # 训练 = 其余原始 + 所有 src 不在测试源的增强样本
    aug_idx = np.where(is_aug == 1)[0]
    train_aug_idx = np.array([i for i in aug_idx if src.to_numpy()[i] not in test_src])
    train = np.concatenate([tr_ori, train_aug_idx])
    test = te_ori
    return train, test, test_src

def main():
    print("=" * 60, flush=True)
    import platform, torch
    print("  python:", platform.python_version())
    print("  torch:", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)

    csv = fetch()
    df = pd.read_csv(csv)
    cats = sorted(df["category"].unique())
    m = {c: i for i, c in enumerate(cats)}
    df["_y"] = df["category"].map(m)
    y = df["_y"].to_numpy()
    texts = df["text"].astype(str).tolist()
    train, test, test_src = build_split(df)
    print(f"训练 {len(train)}（原始 {int((df['is_aug'].to_numpy()[train]==0).sum())} "
          f"+ 增强 {int((df['is_aug'].to_numpy()[train]==1).sum())}）"
          f" / 测试 {len(test)}（全原始）", flush=True)
    print(f"测试源 {len(test_src)} 个，其增强变体全部排除在训练外", flush=True)
    ytr, yte = y[train], y[test]

    rows = []
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- A. BERT 微调 ----
    try:
        from transformers import (BertTokenizerFast, BertForSequenceClassification,
                                  TrainingArguments, Trainer)
        import torch
        from torch.utils.data import Dataset
        tok = BertTokenizerFast.from_pretrained('bert-base-chinese')

        class DS(Dataset):
            def __init__(self, idx):
                enc = tok([texts[i] for i in idx], max_length=MAX_LEN,
                          truncation=True, padding='max_length', return_tensors='pt')
                self.ids = enc['input_ids']; self.am = enc['attention_mask']
                self.lb = torch.tensor(y[idx], dtype=torch.long)
            def __len__(self): return len(self.lb)
            def __getitem__(self, i):
                return {'input_ids': self.ids[i], 'attention_mask': self.am[i],
                        'labels': self.lb[i]}

        model = BertForSequenceClassification.from_pretrained(
            'bert-base-chinese', num_labels=len(cats))
        args = TrainingArguments(output_dir='/kaggle/working/bert_out',
                                 num_train_epochs=2, per_device_train_batch_size=64,
                                 per_device_eval_batch_size=128, learning_rate=2e-5,
                                 logging_steps=100, seed=SEED, report_to=[],
                                 fp16=torch.cuda.is_available())
        t0 = time.time()
        trainer = Trainer(model=model, args=args,
                          train_dataset=DS(train), eval_dataset=DS(test))
        trainer.train()
        fit_dt = time.time() - t0
        logits = trainer.predict(DS(test)).predictions
        pred = logits.argmax(1)
        f1 = f1_score(yte, pred, average="macro")
        acc = accuracy_score(yte, pred)
        print(f"  A.BERT微调 macroF1={f1:.4f} acc={acc:.4f} 训练={fit_dt:.0f}s", flush=True)
        rows.append(dict(name="A.BERT微调", macro_f1=round(float(f1),4),
                         acc=round(float(acc),4), fit_s=round(fit_dt,1),
                         extra=f"(ep2,{dev})"))
    except Exception as e:
        print("  [A] 失败:", repr(e), flush=True)

    # ---- 共用 tokenizer for B/C ----
    import jieba
    cut = lambda t: jieba.lcut(t)

    # ---- B. TF-IDF + LinearSVC ----
    try:
        t0 = time.time()
        tfidf = TfidfVectorizer(tokenizer=cut, lowercase=False, token_pattern=None, min_df=2)
        Xtr = tfidf.fit_transform([texts[i] for i in train])
        Xte = tfidf.transform([texts[i] for i in test])
        clf = LinearSVC(C=1.0); clf.fit(Xtr, ytr)
        pred = clf.predict(Xte); dt = time.time() - t0
        f1 = f1_score(yte, pred, average="macro"); acc = accuracy_score(yte, pred)
        print(f"  B.TFIDF macroF1={f1:.4f} acc={acc:.4f} 训练={dt:.1f}s", flush=True)
        rows.append(dict(name="B.TFIDF", macro_f1=round(float(f1),4),
                         acc=round(float(acc),4), fit_s=round(dt,2),
                         extra=f"(dim={Xtr.shape[1]})"))
    except Exception as e:
        print("  [B] 失败:", repr(e), flush=True)

    # ---- C. 词频随机嵌入 + LinearSVC ----
    try:
        from collections import Counter
        import re
        _P = re.compile(r"^[\W_]+$", re.UNICODE)
        t0 = time.time()
        docs = [[w.strip() for w in cut(t) if w.strip()] for t in texts]
        freq = Counter()
        for d in docs: freq.update(d)
        items = [(w, c) for w, c in freq.items() if c >= 2 and not _P.match(w)]
        items.sort(key=lambda x: (-x[1], x[0]))
        stoi = {w: i for i, w in enumerate(["<pad>", "<unk>"] + [w for w, _ in items])}
        tid = np.zeros((len(docs), MAX_LEN), dtype="int64")
        for i, d in enumerate(docs):
            seq = [stoi.get(w, 1) for w in d][:MAX_LEN]
            tid[i, :len(seq)] = seq
        V = len(stoi)
        rng = np.random.default_rng(SEED)
        W = rng.standard_normal((V, 256)).astype("float32")
        mask = (tid != 0).astype("float32")[..., None]
        E = W[tid] * mask
        Xe = E.sum(1) / mask.sum(1).clip(min=1)
        clf = LinearSVC(C=1.0); clf.fit(Xe[train], ytr)
        pred = clf.predict(Xe[test]); dt = time.time() - t0
        f1 = f1_score(yte, pred, average="macro"); acc = accuracy_score(yte, pred)
        print(f"  C.词频Emb macroF1={f1:.4f} acc={acc:.4f} 训练={dt:.1f}s", flush=True)
        rows.append(dict(name="C.词频Emb", macro_f1=round(float(f1),4),
                         acc=round(float(acc),4), fit_s=round(dt,2),
                         extra=f"(词表{V})"))
    except Exception as e:
        print("  [C] 失败:", repr(e), flush=True)

    rows_sorted = sorted(rows, key=lambda r: -r["macro_f1"])
    print("\n===RESULT_START===")
    print(json.dumps({"rows": rows_sorted, "best": rows_sorted[0] if rows_sorted else None,
                      "split": {"train": int(len(train)), "test": int(len(test)),
                                "test_sources": int(len(test_src))}},
                     ensure_ascii=False))
    print("===RESULT_END===")
    json.dump({"rows": rows_sorted, "best": rows_sorted[0] if rows_sorted else None},
              open('/kaggle/working/result.json','w',encoding='utf-8'),
              ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
