# -*- coding: utf-8 -*-
"""BERT 打榜模型（bert_epoch59.pt）在无泄漏测试集上的 18 阶混淆矩阵。

流程：GitHub Release 下权重 -> /tmp -> 复用 compare_no_leak.py 的无泄漏分组划分
（测试=1378 全原始）-> 逐批评分 -> 输出 18x18 计数混淆矩阵 + 逐类指标。

大文件一律写 /tmp（不进 output 打包），只把一张小 JSON 放 /kaggle/working。
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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification

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

CKPT_DST = "/tmp/" + CKPT        # 大文件放 /tmp，避免被 kernels output 整包回传
CSV_DST = "/tmp/train_augmented.csv"


def download(urls, dst, min_size):
    """带 Range 断点续传的多源下载。"""
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
                got = 0
                with open(dst, mode) as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        got += len(chunk)
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
    print("python", sys.version.split()[0], "| torch", torch.__version__,
          "| cuda", torch.cuda.is_available(), flush=True)

    ck = download(CKPT_URLS, CKPT_DST, 300 * 1024 * 1024)
    print("  权重:", ck, round(os.path.getsize(ck) / 1048576, 1), "MB", flush=True)
    csv = download([RAW_URL, MIRROR_URL], CSV_DST, 1_000_000)

    df = pd.read_csv(csv)
    cats = sorted(df["category"].unique())
    m = {c: i for i, c in enumerate(cats)}
    y = df["category"].map(m).to_numpy()
    texts = df["text"].astype(str).tolist()
    train, test, test_src = build_split(df)
    yte = y[test]
    print("类别数", len(cats), "| 训练", len(train), "| 测试", len(test),
          "| 测试源", len(test_src), flush=True)

    global tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    test_ds = TxtDs([texts[i] for i in test], yte)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]
    n_labels = int(ckpt.get("num_labels", len(cats)))
    print("  权重元信息 epoch=%s test_macro_f1=%s"
          % (ckpt.get("epoch"), ckpt.get("test_macro_f1")), flush=True)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=n_labels)
    model.load_state_dict(state)
    model.to(device).eval()
    print("  模型就绪 device=", device, flush=True)

    t0 = time.time()
    preds, gts = [], []
    with torch.no_grad():
        for b in test_loader:
            logits = model(input_ids=b["input_ids"].to(device),
                           attention_mask=b["attention_mask"].to(device)).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(b["labels"].tolist())
    dt = time.time() - t0

    cm = confusion_matrix(gts, preds, labels=list(range(len(cats))))
    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds, average="macro")
    print("推理 %.1fs | acc=%.4f macroF1=%.4f" % (dt, acc, f1), flush=True)

    out = {
        "name": "BERT微调(抽样1024,ep59)",
        "labels": cats,
        "matrix": cm.tolist(),
        "acc": round(float(acc), 4),
        "macro_f1": round(float(f1), 4),
        "n_test": int(len(gts)),
        "row_sum": cm.sum(axis=1).tolist(),
        "infer_s": round(dt, 1),
    }
    json.dump(out, open("/kaggle/working/bert_confusion.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("===RESULT_START===")
    print(json.dumps(out, ensure_ascii=False))
    print("===RESULT_END===")
    print("done.", flush=True)


if __name__ == "__main__":
    main()
