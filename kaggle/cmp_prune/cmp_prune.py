# -*- coding: utf-8 -*-
"""非结构化剪枝单独内核：30%/50% 全局 L1 剪枝 + 实测稀疏度 + 评估。"""
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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizerFast, BertForSequenceClassification

SEED = 42
MAX_LEN = 128
MODEL_PATH = "bert-base-chinese"
OUT_NAME = "bert_cmp_prune.json"

CKPT = "bert_epoch59.pt"
CKPT_URLS = [
    "https://github.com/Barry-Wuu/text_classfication/releases/download/bert-weights/" + CKPT,
    "https://ghfast.top/https://github.com/Barry-Wuu/text_classfication/releases/download/bert-weights/" + CKPT,
]
RAW_URL = "https://raw.githubusercontent.com/Barry-Wuu/text_classfication/main/train_augmented.csv"
MIRROR_URL = "https://ghfast.top/" + RAW_URL
CKPT_DST = "/tmp/" + CKPT
CSV_DST = "/tmp/train_augmented.csv"

RESULTS = {}
DEV_GPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEV_CPU = torch.device("cpu")


def download(urls, dst, min_size):
    if os.path.exists(dst) and os.path.getsize(dst) >= min_size:
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
                    return dst
                raise IOError("size too small")
            except Exception as e:
                last = e
                print("  下载重试 %s err=%s" % (url[:50], str(e)[:80]), flush=True)
                time.sleep(3)
    raise RuntimeError("下载失败: %r" % (last,))


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
    return np.concatenate([tr_ori, train_aug_idx]), te_ori


class TxtDs(Dataset):
    def __init__(self, texts, labels):
        enc = tokenizer(list(texts), max_length=MAX_LEN, truncation=True,
                        padding="max_length", return_tensors="pt")
        self.ids = enc["input_ids"]; self.am = enc["attention_mask"]
        self.lb = torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.lb)

    def __getitem__(self, i):
        return {"input_ids": self.ids[i], "attention_mask": self.am[i],
                "labels": self.lb[i]}


def forward_logits(model, ids, att):
    out = model(input_ids=ids, attention_mask=att)
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, tuple):
        return out[0]
    return out


def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for b in loader:
            logits = forward_logits(model, b["input_ids"].to(device),
                                    b["attention_mask"].to(device))
            preds.extend(logits.argmax(-1).cpu().tolist())
            gts.extend(b["labels"].tolist())
    return preds, gts


def latency(model, loader, device, repeats=3):
    is_gpu = device.type == "cuda"

    def one_pass():
        t0 = time.perf_counter()
        with torch.no_grad():
            for b in loader:
                forward_logits(model, b["input_ids"].to(device),
                               b["attention_mask"].to(device))
        if is_gpu:
            torch.cuda.synchronize()
        return time.perf_counter() - t0

    one_pass()
    ts = sorted(one_pass() for _ in range(repeats))
    return ts[len(ts) // 2] / N_TEST * 1000.0


def size_mb(model):
    tmp = "/tmp/_size_probe.pt"
    torch.save(model.state_dict(), tmp)
    mb = os.path.getsize(tmp) / 1048576.0
    os.remove(tmp)
    return round(mb, 1)


def record(name, model=None, dev=None, preds=None, gts=None, note=""):
    m = {}
    if preds is not None:
        m["acc"] = round(float(accuracy_score(gts, preds)), 4)
        m["macro_f1"] = round(float(f1_score(gts, preds, average="macro")), 4)
    if model is not None:
        m["size_mb"] = size_mb(model)
    if dev is not None:
        m["ms_per_sample"] = round(latency(model, test_loader, dev), 2)
    m["note"] = note
    RESULTS[name] = m
    print(f"[record] {name}: {json.dumps(m, ensure_ascii=False)}", flush=True)


def load_teacher(n_labels):
    teacher = BertForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=n_labels)
    ckpt = torch.load(CKPT_DST, map_location="cpu", weights_only=False)
    teacher.load_state_dict(ckpt["state_dict"])
    return teacher


def main():
    print("python", sys.version.split()[0], "| torch", torch.__version__,
          "| cuda", torch.cuda.is_available(),
          "| gpu", torch.cuda.get_device_name(0) if DEV_GPU.type == "cuda" else "-",
          flush=True)

    download(CKPT_URLS, CKPT_DST, 300 * 1024 * 1024)
    download([RAW_URL, MIRROR_URL], CSV_DST, 1_000_000)

    global tokenizer, test_loader, N_TEST
    df = pd.read_csv(CSV_DST)
    cats = sorted(df["category"].unique())
    m = {c: i for i, c in enumerate(cats)}
    y = df["category"].map(m).to_numpy()
    texts = df["text"].astype(str).tolist()
    train, test = build_split(df)
    yte = y[test]
    N_TEST = len(test)
    print("类别数", len(cats), "| 训练", len(train), "| 测试", N_TEST, flush=True)

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    test_ds = TxtDs([texts[i] for i in test], yte)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    from torch.nn.utils import prune
    ckpt_state = torch.load(CKPT_DST, map_location="cpu", weights_only=False)["state_dict"]
    for amount in (0.3, 0.5):
        p = BertForSequenceClassification.from_pretrained(
            MODEL_PATH, num_labels=len(cats))
        p.load_state_dict(ckpt_state)
        params = []
        for name, mod in p.named_modules():
            if isinstance(mod, nn.Linear) and "classifier" not in name:
                params.append((mod, "weight"))
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured,
                                  amount=amount)
        for mod, _ in params:
            prune.remove(mod, "weight")
        p.to(DEV_GPU).eval()
        # 剪枝是否真实发生的证据：权重零元素占比
        zeros = tot = 0
        for mod, _ in params:
            zeros += int((mod.weight == 0).sum()); tot += mod.weight.numel()
        sparsity = zeros / tot
        print("  sparsity=%.4f (target %.2f)" % (sparsity, amount), flush=True)
        preds, gts = evaluate(p, test_loader, DEV_GPU)
        record("剪枝%d%%(非结构化)" % int(amount * 100), p, DEV_GPU, preds, gts,
               note="global_unstructured L1, 实测稀疏度%.3f, 逻辑剪枝体积不变" % sparsity)
        if amount == 0.3:
            record("剪枝30%%(CPU)", p.cpu(), DEV_CPU, note="CPU延迟对照")


    json.dump({"n_test": N_TEST, "batch": 128, "max_len": MAX_LEN,
               "gpu": torch.cuda.get_device_name(0) if DEV_GPU.type == "cuda" else "cpu",
               "results": RESULTS},
              open("/kaggle/working/" + OUT_NAME, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("===RESULT_START===")
    print(json.dumps(RESULTS, ensure_ascii=False))
    print("===RESULT_END===")
    print("done.", flush=True)


if __name__ == "__main__":
    main()
