# -*- coding: utf-8 -*-
"""生成 4 个独立压缩内核（每方法一个）。"""
import os
import json

BASE = r"D:\B++\WorkBuddy\kaggle_bert_compress"

COMMON = '''# -*- coding: utf-8 -*-
"""{doc}"""
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
from transformers import BertTokenizerFast, BertForSequenceClassification{extra_imports}

SEED = 42
MAX_LEN = 128
MODEL_PATH = "bert-base-chinese"
OUT_NAME = "{out_name}"

CKPT = "bert_epoch59.pt"
CKPT_URLS = [
    "https://github.com/Barry-Wuu/text_classfication/releases/download/bert-weights/" + CKPT,
    "https://ghfast.top/https://github.com/Barry-Wuu/text_classfication/releases/download/bert-weights/" + CKPT,
]
RAW_URL = "https://raw.githubusercontent.com/Barry-Wuu/text_classfication/main/train_augmented.csv"
MIRROR_URL = "https://ghfast.top/" + RAW_URL
CKPT_DST = "/tmp/" + CKPT
CSV_DST = "/tmp/train_augmented.csv"

RESULTS = {{}}
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
                h = {{"Range": "bytes=%d-" % pos}} if pos else {{}}
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
        return {{"input_ids": self.ids[i], "attention_mask": self.am[i],
                "labels": self.lb[i]}}


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
    m = {{}}
    if preds is not None:
        m["acc"] = round(float(accuracy_score(gts, preds)), 4)
        m["macro_f1"] = round(float(f1_score(gts, preds, average="macro")), 4)
    if model is not None:
        m["size_mb"] = size_mb(model)
    if dev is not None:
        m["ms_per_sample"] = round(latency(model, test_loader, dev), 2)
    m["note"] = note
    RESULTS[name] = m
    print(f"[record] {{name}}: {{json.dumps(m, ensure_ascii=False)}}", flush=True)


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
    m = {{c: i for i, c in enumerate(cats)}}
    y = df["category"].map(m).to_numpy()
    texts = df["text"].astype(str).tolist()
    train, test = build_split(df)
    yte = y[test]
    N_TEST = len(test)
    print("类别数", len(cats), "| 训练", len(train), "| 测试", N_TEST, flush=True)

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    test_ds = TxtDs([texts[i] for i in test], yte)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

{experiment}

    json.dump({{"n_test": N_TEST, "batch": 128, "max_len": MAX_LEN,
               "gpu": torch.cuda.get_device_name(0) if DEV_GPU.type == "cuda" else "cpu",
               "results": RESULTS}},
              open("/kaggle/working/" + OUT_NAME, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("===RESULT_START===")
    print(json.dumps(RESULTS, ensure_ascii=False))
    print("===RESULT_END===")
    print("done.", flush=True)


if __name__ == "__main__":
    main()
'''

INT8_EXP = '''    # ---- 教师基准（CPU） ----
    teacher = load_teacher(len(cats)).to(DEV_CPU).eval()
    preds, gts = evaluate(teacher, test_loader, DEV_CPU)
    record("教师BERT fp32-CPU", teacher, DEV_CPU, preds, gts, note="基准(CPU)")

    # ---- INT8 动态量化（DQ，讲义口径：CPU） ----
    q = torch.ao.quantization.quantize_dynamic(
        teacher, {nn.Linear}, dtype=torch.qint8)
    q.eval()
    preds, gts = evaluate(q, test_loader, DEV_CPU)
    record("INT8动态量化(CPU)", q, DEV_CPU, preds, gts,
           note="quantize_dynamic Linear->qint8")
'''

NF4_EXP = '''    try:
        import bitsandbytes as bnb
    except ImportError:
        import subprocess as sp
        sp.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"],
               check=True)
        import bitsandbytes as bnb
    print("bnb", bnb.__version__, flush=True)
    # ---- 教师基准（GPU） ----
    teacher = load_teacher(len(cats)).to(DEV_GPU).eval()
    preds, gts = evaluate(teacher, test_loader, DEV_GPU)
    record("教师BERT fp32", teacher, DEV_GPU, preds, gts, note="基准(GPU)")

    # ---- NF4 4bit（bnb，encoder Linear 换 4bit） ----
    m4 = BertForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=len(cats))
    state = torch.load(CKPT_DST, map_location="cpu", weights_only=False)["state_dict"]
    for name, module in list(m4.named_modules()):
        if isinstance(module, nn.Linear) and ".encoder." in name + ".":
            parent = m4.get_submodule(name.rsplit(".", 1)[0])
            leaf = name.rsplit(".", 1)[1]
            setattr(parent, leaf, bnb.nn.Linear4bit(
                module.in_features, module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.float16, quant_type="nf4"))
    m4.load_state_dict(state, strict=False)
    m4.to(DEV_GPU).eval()
    preds, gts = evaluate(m4, test_loader, DEV_GPU)
    record("NF4 4bit量化(GPU)", m4, DEV_GPU, preds, gts,
           note="bnb NF4, encoder Linear 4bit")
'''

PRUNE_EXP = '''    from torch.nn.utils import prune
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
'''

KD_EXP = '''    T, ALPHA, EPOCHS, PER_EPOCH, BATCH = 2, 0.7, 12, 1024, 32

    class LstmStudent(nn.Module):
        """讲义口径的蒸馏学生：Embedding + BiLSTM + Linear。"""

        def __init__(self, vocab, n_cls, emb=128, hid=128, layers=1):
            super().__init__()
            self.emb = nn.Embedding(vocab, emb, padding_idx=0)
            self.lstm = nn.LSTM(emb, hid, num_layers=layers, batch_first=True,
                                bidirectional=True)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(2 * hid, n_cls)

        def forward(self, input_ids, attention_mask=None):
            x = self.emb(input_ids)
            lengths = (input_ids != 0).sum(1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False)
            _, (h, _) = self.lstm(packed)
            h = torch.cat([h[-2], h[-1]], dim=-1)
            return self.fc(self.drop(h))

    # ---- 教师基准（GPU，作软标签来源与校验） ----
    teacher = load_teacher(len(cats)).to(DEV_GPU).eval()
    preds, gts = evaluate(teacher, test_loader, DEV_GPU)
    record("教师BERT fp32", teacher, DEV_GPU, preds, gts, note="基准(GPU)")

    # ---- LSTM 学生：软标签蒸馏（讲义口径 T=2, α=0.7） ----
    student = LstmStudent(tokenizer.vocab_size, len(cats)).to(DEV_GPU)
    print("学生参数量 %.2fM" % (sum(p.numel() for p in student.parameters()) / 1e6),
          flush=True)
    optim = torch.optim.AdamW(student.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    rng = np.random.RandomState(SEED)
    train_texts = [texts[i] for i in train]
    ytrain = y[train]
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        idx = rng.choice(len(train_texts), size=min(PER_EPOCH, len(train_texts)),
                         replace=False)
        ep_texts = [train_texts[i] for i in idx]
        ep_labels = ytrain[idx]
        for s in range(0, len(ep_texts), BATCH):
            bt = ep_texts[s:s + BATCH]
            bl = torch.tensor(ep_labels[s:s + BATCH], device=DEV_GPU)
            enc = tokenizer(bt, max_length=MAX_LEN, truncation=True,
                            padding=True, return_tensors="pt")
            ids, att = enc["input_ids"].to(DEV_GPU), enc["attention_mask"].to(DEV_GPU)
            with torch.no_grad():
                t_logits = forward_logits(teacher, ids, att)
            s_logits = student(input_ids=ids)
            hard = ce(s_logits, bl)
            soft = kl(torch.log_softmax(s_logits / T, -1),
                      torch.softmax(t_logits / T, -1)) * (T * T)
            loss = (1 - ALPHA) * hard + ALPHA * soft
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optim.step()
        print("  KD epoch %d/%d loss=%.4f (%.0fs)"
              % (ep, EPOCHS, loss.item(), time.time() - t0), flush=True)
    kd_train_s = round(time.time() - t0, 1)
    student.eval()
    preds, gts = evaluate(student, test_loader, DEV_GPU)
    record("蒸馏LSTM学生 fp32", student, DEV_GPU, preds, gts,
           note="T=2 a=0.7 12轮x1024 抽样, GPU延迟")
    RESULTS["蒸馏LSTM学生 fp32"]["train_s"] = kd_train_s
    student_cpu = LstmStudent(tokenizer.vocab_size, len(cats))
    student_cpu.load_state_dict(student.state_dict()); student_cpu.eval()
    record("蒸馏LSTM学生 fp32-CPU", student_cpu, DEV_CPU, note="CPU延迟对照")
    q = torch.ao.quantization.quantize_dynamic(
        student_cpu, {nn.Linear, nn.LSTM}, dtype=torch.qint8)
    q.eval()
    preds, gts = evaluate(q, test_loader, DEV_CPU)
    record("蒸馏LSTM学生 INT8(CPU)", q, DEV_CPU, preds, gts,
           note="LSTM/Linear 量化, Embedding 保留 fp32")
'''

KD2_EXP = '''    T, ALPHA, EPOCHS, BATCH = 2, 0.7, 8, 128

    class LstmStudent(nn.Module):
        """讲义口径的蒸馏学生：Embedding + BiLSTM + Linear。"""

        def __init__(self, vocab, n_cls, emb=128, hid=128, layers=1):
            super().__init__()
            self.emb = nn.Embedding(vocab, emb, padding_idx=0)
            self.lstm = nn.LSTM(emb, hid, num_layers=layers, batch_first=True,
                                bidirectional=True)
            self.drop = nn.Dropout(0.3)
            self.fc = nn.Linear(2 * hid, n_cls)

        def forward(self, input_ids, attention_mask=None):
            x = self.emb(input_ids)
            lengths = (input_ids != 0).sum(1).clamp(min=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False)
            _, (h, _) = self.lstm(packed)
            h = torch.cat([h[-2], h[-1]], dim=-1)
            return self.fc(self.drop(h))

    teacher = load_teacher(len(cats)).to(DEV_GPU).eval()
    preds, gts = evaluate(teacher, test_loader, DEV_GPU)
    record("教师BERT fp32", teacher, DEV_GPU, preds, gts, note="基准(GPU)")

    train_texts = [texts[i] for i in train]
    ytrain = y[train]
    # 训练集一次性 tokenize
    t0 = time.time()
    enc = tokenizer(train_texts, max_length=MAX_LEN, truncation=True,
                    padding="max_length", return_tensors="pt")
    tr_ids, tr_att = enc["input_ids"], enc["attention_mask"]
    print("训练集 tokenize %.0fs" % (time.time() - t0), flush=True)
    # 教师软标签预计算（全训练集一次前向，避免每轮重复）
    t0 = time.time()
    chunks = []
    with torch.no_grad():
        for s in range(0, len(train_texts), 256):
            chunks.append(forward_logits(
                teacher, tr_ids[s:s + 256].to(DEV_GPU),
                tr_att[s:s + 256].to(DEV_GPU)).cpu())
    soft_all = torch.cat(chunks)
    print("教师软标签预计算 %.0fs shape=%s" % (time.time() - t0, tuple(soft_all.shape)),
          flush=True)

    student = LstmStudent(tokenizer.vocab_size, len(cats)).to(DEV_GPU)
    print("学生参数量 %.2fM" % (sum(p.numel() for p in student.parameters()) / 1e6),
          flush=True)
    optim = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    ytrain_t = torch.tensor(ytrain, dtype=torch.long)
    rng = np.random.RandomState(SEED)
    t0 = time.time()
    for ep in range(1, EPOCHS + 1):
        order = rng.permutation(len(train_texts))
        tot_loss = 0.0
        nb = 0
        for s in range(0, len(order), BATCH):
            idx = torch.tensor(order[s:s + BATCH])
            ids = tr_ids[idx].to(DEV_GPU)
            bl = ytrain_t[idx].to(DEV_GPU)
            t_logits = soft_all[idx].to(DEV_GPU)
            s_logits = student(input_ids=ids)
            hard = ce(s_logits, bl)
            soft = kl(torch.log_softmax(s_logits / T, -1),
                      torch.softmax(t_logits / T, -1)) * (T * T)
            loss = (1 - ALPHA) * hard + ALPHA * soft
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optim.step()
            tot_loss += loss.item(); nb += 1
        sched.step()
        print("  KD epoch %d/%d loss=%.4f (%.0fs)"
              % (ep, EPOCHS, tot_loss / max(nb, 1), time.time() - t0), flush=True)
    kd_train_s = round(time.time() - t0, 1)
    student.eval()
    preds, gts = evaluate(student, test_loader, DEV_GPU)
    record("蒸馏LSTM学生 fp32", student, DEV_GPU, preds, gts,
           note="T=2 a=0.7 全量%d条x%d轮, GPU延迟" % (len(train_texts), EPOCHS))
    RESULTS["蒸馏LSTM学生 fp32"]["train_s"] = kd_train_s
    student_cpu = LstmStudent(tokenizer.vocab_size, len(cats))
    student_cpu.load_state_dict(student.state_dict()); student_cpu.eval()
    record("蒸馏LSTM学生 fp32-CPU", student_cpu, DEV_CPU, note="CPU延迟对照")
    q = torch.ao.quantization.quantize_dynamic(
        student_cpu, {nn.Linear, nn.LSTM}, dtype=torch.qint8)
    q.eval()
    preds, gts = evaluate(q, test_loader, DEV_CPU)
    record("蒸馏LSTM学生 INT8(CPU)", q, DEV_CPU, preds, gts,
           note="LSTM/Linear 量化, Embedding 保留 fp32")
'''

KD3_EXP = KD2_EXP.replace('T, ALPHA, EPOCHS, BATCH = 2, 0.7, 8, 128',
                          'T, ALPHA, EPOCHS, BATCH = 2, 0.7, 20, 128'
                          ).replace('def __init__(self, vocab, n_cls, emb=128, hid=128, layers=1):',
                                    'def __init__(self, vocab, n_cls, emb=256, hid=256, layers=1):'
                          ).replace('self.drop = nn.Dropout(0.3)', 'self.drop = nn.Dropout(0.2)'
                          ).replace('lr=1e-3, weight_decay=0.01', 'lr=1.5e-3, weight_decay=0.01'
                          ).replace('全量%d条x%d轮', '全量%d条x%d轮(256维学生)')

JOBS = [
    ("kaggle_cmp_int8", "textcls-cmp-int8", "textcls cmp int8", "cmp_int8.py",
     "INT8 动态量化（DQ）单独内核：教师 CPU 基准 + INT8 评估/体积/延迟。", "", "bert_cmp_int8.json", False, INT8_EXP),
    ("kaggle_cmp_nf4", "textcls-cmp-nf4", "textcls cmp nf4", "cmp_nf4.py",
     "NF4 4bit 量化单独内核：教师 GPU 基准 + bnb NF4 评估/体积/延迟。", "", "bert_cmp_nf4.json", True, NF4_EXP),
    ("kaggle_cmp_prune", "textcls-cmp-prune", "textcls cmp prune", "cmp_prune.py",
     "非结构化剪枝单独内核：30%/50% 全局 L1 剪枝 + 实测稀疏度 + 评估。", "", "bert_cmp_prune.json", True, PRUNE_EXP),
    ("kaggle_cmp_kd", "textcls-cmp-kd", "textcls cmp kd", "cmp_kd.py",
     "软标签蒸馏单独内核：LSTM 学生（讲义口径），T=2 α=0.7。", "", "bert_cmp_kd.json", True, KD_EXP),
    ("kaggle_cmp_kd2", "textcls-cmp-kd2", "textcls cmp kd2", "cmp_kd2.py",
     "软标签蒸馏 v2：LSTM 学生全量训练 8 轮 + 教师软标签预缓存（修 v1 训练不足）。",
     "", "bert_cmp_kd2.json", True, KD2_EXP),
    ("kaggle_cmp_kd3", "textcls-cmp-kd3", "textcls cmp kd3", "cmp_kd3.py",
     "软标签蒸馏 v3：256 维 BiLSTM 学生 + 20 轮全量（探训练充分度的上限）。",
     "", "bert_cmp_kd3.json", True, KD3_EXP),
]

for d, slug, title, code, doc, extra, out, gpu, exp in JOBS:
    dd = os.path.join(BASE, d)
    os.makedirs(dd, exist_ok=True)
    script = COMMON.format(doc=doc, extra_imports=extra, out_name=out,
                           experiment=exp)
    open(os.path.join(dd, code), "w", encoding="utf-8", newline="\n").write(script)
    meta = {"id": "wubarry/" + slug, "title": title, "code_file": code,
            "language": "python", "kernel_type": "script", "is_private": True,
            "enable_gpu": gpu, "enable_internet": True, "enable_tpu": False,
            "keywords": [], "dataset_sources": [], "kernel_sources": [],
            "competition_sources": [], "model_sources": []}
    json.dump(meta, open(os.path.join(dd, "kernel-metadata.json"), "w",
                         encoding="utf-8"), indent=1, ensure_ascii=False)
    print("written", dd)
# 语法快检
import py_compile, glob
for d, *_ in JOBS:
    for f in glob.glob(os.path.join(BASE, d, "*.py")):
        py_compile.compile(f, doraise=True)
print("syntax OK")
