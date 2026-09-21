# -*- coding: utf-8 -*-
"""张量方案 A：BERT tokenizer（bert-base-chinese）。

把 labeled.csv 的 text 编码为固定长度 200 的整数张量：
  input_ids / attention_mask / token_type_ids  (int32, shape=[N,200])
  labels                                        (int64, shape=[N])

产物：resources/tensors_bert.npz  （np.load 即用）
      resources/bert_token_lengths.npy （未截断前的真实长度，便于分析）
"""
import os
import time
import numpy as np
import pandas as pd
from transformers import BertTokenizerFast

ROOT = r"D:\text_classfication"
MODEL_DIR = r"D:\models\bert-base-chinese"
MAX_LEN = 200
BATCH = 256

def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    df = pd.read_csv(os.path.join(ROOT, "labeled.csv"))
    texts = df["text"].astype(str).tolist()
    labels = df["label"].to_numpy(dtype="int64")

    tok = BertTokenizerFast.from_pretrained(MODEL_DIR)
    n = len(texts)
    input_ids = np.zeros((n, MAX_LEN), dtype="int32")
    attn = np.zeros((n, MAX_LEN), dtype="int32")
    tok_type = np.zeros((n, MAX_LEN), dtype="int32")
    raw_len = np.zeros(n, dtype="int32")

    t0 = time.time()
    for start in range(0, n, BATCH):
        chunk = texts[start:start + BATCH]
        # 先用 truncation-only 拿"截断前的真实长度"
        raw = tok(chunk, truncation=True, max_length=10**9)["input_ids"]
        raw_len[start:start + len(chunk)] = np.array([len(x) for x in raw], dtype="int32")
        # 再编码为固定长度张量
        enc = tok(chunk, max_length=MAX_LEN, truncation=True,
                  padding="max_length")
        m = len(chunk)
        input_ids[start:start + m] = np.asarray(enc["input_ids"], dtype="int32")
        attn[start:start + m] = np.asarray(enc["attention_mask"], dtype="int32")
        tok_type[start:start + m] = np.asarray(enc["token_type_ids"], dtype="int32")
    dt = time.time() - t0

    np.savez_compressed(
        os.path.join(ROOT, "resources", "tensors_bert.npz"),
        input_ids=input_ids, attention_mask=attn,
        token_type_ids=tok_type, labels=labels,
    )
    np.save(os.path.join(ROOT, "resources", "bert_token_lengths.npy"), raw_len)

    truncated = int((raw_len > MAX_LEN).sum())
    print(f"[OK] BERT 张量: shape={input_ids.shape}  耗时={dt:.1f}s")
    print(f"     词表大小={tok.vocab_size}  截断样本={truncated} ({truncated/n*100:.2f}%)")
    print(f"     raw_len P50={int(np.percentile(raw_len,50))} "
          f"P95={int(np.percentile(raw_len,95))} P99={int(np.percentile(raw_len,99))} "
          f"max={int(raw_len.max())}")
    print(f"     npz 体积={os.path.getsize(os.path.join(ROOT,'resources','tensors_bert.npz'))/1e6:.1f} MB")

if __name__ == "__main__":
    main()
