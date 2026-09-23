# -*- coding: utf-8 -*-
"""逐样本联合分布：LinearSVC 判对 / BERT(epoch59) 判对 的交集，供韦恩图用。

流程：下载数据与权重 -> 无泄漏分组划分（与其它脚本同参同种子）->
词级 TF-IDF + LinearSVC(C=1.0) 预测 -> BERT CPU 推理 -> 逐类联合计数。
输出 /kaggle/working/bert_svc_joint.json
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
    txt = ''.join(traceback.format_exception(t, v, tb))
    print('FATAL: ' + txt, flush=True)
    try: open('/kaggle/working/FATAL.txt', 'w', encoding='utf-8').write(txt)
    except Exception: pass


sys.excepthook = _exchook

os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

import numpy as np
import pandas as pd
import requests
import jieba
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification

jieba.setLogLevel(20)
SEED = 42
MAX_LEN = 128
MODEL_PATH = "bert-base-chinese"

CKPT = "bert_epoch59.pt"
CKPT_URLS = [
    "https://github.com/Barry-Wuu/text_classfication/releases/download/bert-weights/" + CKPT,
    "https://ghfast.top/https://github.com/Barry-Wuu/text_classfication/releases/download/bert-weights/" + CKPT,
]
RAW_URL = "https://raw.githubusercontent.com/Barry-Wuu/text_classfication/main/train_augmented.csv"
MIRROR_URL = "https://ghfast.top/" + RAW_URL

CKPT_DST = "/tmp/" + CKPT
CSV_DST = "/tmp/train_augmented.csv"


def download(urls, dst, min_size):
    if os.path.exists(dst) and os.path.getsize(dst) >= min_size:
        print("  已存在:", dst, os.path.getsize(dst), flush=True)
        return dst
    last = None
    for url in urls:
        pos = os.path.getsize(dst) if os.path.exists(dst) else 0
        for attempt in range(4):
            try:
                h = {"Range": "bytes=%d-" % pos} if pos else {}
                r = requests.get(url, headers=h, stream=True, timeout=300)
                if r.status_code not in (200, 206):
                    raise IOError("HTTP %s" % r.status_code)
                mode = "ab" if (pos and r.status_code == 206) else "wb"
                if mode == "wb":
                    pos = 0
                with open(dst, mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        pos += len(chunk)
                if os.path.getsize(dst) >= min_size:
                    print("  下载完成:", os.path.getsize(dst), "字节", flush=True)
                    return dst
                raise IOError("size %d < %d" % (os.path.getsize(dst), min_size))
            except Exception as e:
                last = e
                print("  下载重试 source=%s try=%d pos=%d err=%s"
                      % (url[:60], attempt, pos, str(e)[:90]), flush=True)
                time.sleep(3 + 4 * attempt)
    raise RuntimeError("全部下载源失败: %r" % (last,))


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


class TxtDs(Dataset):
    def __init__(self, texts, labels):
        enc = tokenizer(list(texts), max_length=MAX_LEN, truncation=True,
                        padding="max_length", return_tensors="pt")
        self.ids = enc["input_ids"]; self.am = enc["attention_mask"]
        self.lb = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.lb)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.am[i], "labels": self.lb[i]}


def main():
    print("python", sys.version.split()[0], "| torch", torch.__version__, flush=True)

    ck = download(CKPT_URLS, CKPT_DST, 300 * 1024 * 1024)
    csv = download([RAW_URL, MIRROR_URL], CSV_DST, 1_000_000)

    df = pd.read_csv(csv)
    cats = sorted(df["category"].unique())
    m = {c: i for i, c in enumerate(cats)}
    y = df["category"].map(m).to_numpy()
    texts = df["text"].astype(str).tolist()
    train, test, test_src = build_split(df)
    yte = y[test]
    print("类别数", len(cats), "| 训练", len(train), "| 测试", len(test), flush=True)

    # ---- LinearSVC（与 compare_tfidf_clf.py 同参） ----
    t0 = time.time()
    tf = TfidfVectorizer(tokenizer=jieba.lcut, lowercase=False,
                         token_pattern=None, min_df=2)
    Xtr = tf.fit_transform([texts[i] for i in train])
    Xte = tf.transform([texts[i] for i in test])
    svc = LinearSVC(C=1.0)
    svc.fit(Xtr, y[train])
    svc_pred = svc.predict(Xte)
    print("LinearSVC acc=%.4f (%.1fs)" % (accuracy_score(yte, svc_pred),
                                          time.time() - t0), flush=True)

    # ---- BERT CPU 推理 ----
    global tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    test_ds = TxtDs([texts[i] for i in test], yte)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    n_labels = int(ckpt.get("num_labels", len(cats)))
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=n_labels)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print("  模型就绪", flush=True)

    t0 = time.time()
    bert_pred = []
    with torch.no_grad():
        for b in test_loader:
            logits = model(input_ids=b["input_ids"],
                           attention_mask=b["attention_mask"]).logits
            bert_pred.extend(logits.argmax(-1).cpu().tolist())
    bert_pred = np.array(bert_pred)
    print("BERT acc=%.4f (%.1fs)" % (accuracy_score(yte, bert_pred),
                                     time.time() - t0), flush=True)

    # ---- 联合分布 ----
    svc_ok = svc_pred == yte
    bert_ok = bert_pred == yte
    joint = {}
    for k, c in enumerate(cats):
        idx = yte == k
        joint[c] = {
            "n": int(idx.sum()),
            "both": int((svc_ok & bert_ok)[idx].sum()),
            "svc_only": int((svc_ok & ~bert_ok)[idx].sum()),
            "bert_only": int((~svc_ok & bert_ok)[idx].sum()),
            "neither": int((~svc_ok & ~bert_ok)[idx].sum()),
        }
    joint["_overall"] = {
        "n": int(len(test)),
        "both": int((svc_ok & bert_ok).sum()),
        "svc_only": int((svc_ok & ~bert_ok).sum()),
        "bert_only": int((~svc_ok & bert_ok).sum()),
        "neither": int((~svc_ok & ~bert_ok).sum()),
        "svc_acc": round(float(svc_ok.mean()), 4),
        "bert_acc": round(float(bert_ok.mean()), 4),
    }
    json.dump(joint, open("/kaggle/working/bert_svc_joint.json", "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
    print("===RESULT_START===")
    print(json.dumps(joint, ensure_ascii=False))
    print("===RESULT_END===")
    print("done.", flush=True)


if __name__ == "__main__":
    main()
