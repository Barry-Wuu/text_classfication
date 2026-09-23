# -*- coding: utf-8 -*-
"""对打榜模型 bert_epoch59.pt 做四种压缩：INT8 动态量化 / NF4 4bit 量化 /
软标签蒸馏(4层学生) / 非结构化剪枝，统一在无泄漏测试集(1378)上评估。

讲义口径：DQ 量化必须 CPU；蒸馏 loss = (1-α)*硬标签CE + α*T²*KL(软标签)，
T=2, α=0.7；剪枝用 torch.nn.utils.prune.global_unstructured（逻辑剪枝，
体积不变是预期行为，要如实报告）。

统一协议：准确率/宏F1 在测试集 1378 条上算；体积 = state_dict 存盘字节数；
延迟 = 1 次预热 + 3 次计时的全测试集推理取中位，报 ms/样本（GPU 计时带
synchronize）。每组实验独立 try/except，单组失败不拖垮整体。
输出 /kaggle/working/bert_compress.json
"""
import sys, os, json, time, traceback, shutil

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
from transformers import BertTokenizerFast, BertForSequenceClassification, BertConfig

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

RESULTS = {}          # name -> metrics dict
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
                print("  下载重试 %s try=%d err=%s" % (url[:50], attempt, str(e)[:80]), flush=True)
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
    return np.concatenate([tr_ori, train_aug_idx]), te_ori, test_src


class TxtDs(Dataset):
    def __init__(self, texts, labels=None):
        enc = tokenizer(list(texts), max_length=MAX_LEN, truncation=True,
                        padding="max_length", return_tensors="pt")
        self.ids = enc["input_ids"]; self.am = enc["attention_mask"]
        self.lb = None if labels is None else torch.tensor(labels, dtype=torch.long)

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        if self.lb is None:
            return {"input_ids": self.ids[i], "attention_mask": self.am[i]}
        return {"input_ids": self.ids[i], "attention_mask": self.am[i],
                "labels": self.lb[i]}


def evaluate(model, loader, device):
    model.eval()
    preds, gts = [], []
    with torch.no_grad():
        for b in loader:
            logits = model(input_ids=b["input_ids"].to(device),
                           attention_mask=b["attention_mask"].to(device)).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
            if "labels" in b:
                gts.extend(b["labels"].tolist())
    return preds, gts


def latency(model, loader, device, repeats=3):
    """全测试集推理计时，返回 ms/样本。GPU 计时带 synchronize。"""
    is_gpu = device.type == "cuda"
    def one_pass():
        t0 = time.perf_counter()
        with torch.no_grad():
            for b in loader:
                model(input_ids=b["input_ids"].to(device),
                      attention_mask=b["attention_mask"].to(device))
        if is_gpu:
            torch.cuda.synchronize()
        return time.perf_counter() - t0
    one_pass()                                   # 预热
    ts = sorted(one_pass() for _ in range(repeats))
    return ts[len(ts) // 2] / N_TEST * 1000.0    # ms/样本


def size_mb(model):
    """state_dict 存盘的实际字节数。"""
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


def main():
    print("python", sys.version.split()[0], "| torch", torch.__version__,
          "| cuda", torch.cuda.is_available(),
          "| gpu", torch.cuda.get_device_name(0) if DEV_GPU.type == "cuda" else "-", flush=True)

    ck = download(CKPT_URLS, CKPT_DST, 300 * 1024 * 1024)
    csv = download([RAW_URL, MIRROR_URL], CSV_DST, 1_000_000)

    global tokenizer, test_loader, N_TEST
    df = pd.read_csv(csv)
    cats = sorted(df["category"].unique())
    m = {c: i for i, c in enumerate(cats)}
    y = df["category"].map(m).to_numpy()
    texts = df["text"].astype(str).tolist()
    train, test, _ = build_split(df)
    yte = y[test]
    N_TEST = len(test)
    print("类别数", len(cats), "| 训练", len(train), "| 测试", N_TEST, flush=True)

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    test_ds = TxtDs([texts[i] for i in test], yte)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
    train_texts = [texts[i] for i in train]
    ytrain = y[train]

    # ============ 教师模型 FP32（基准） ============
    teacher = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(cats))
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    teacher.load_state_dict(ckpt["state_dict"])
    teacher.to(DEV_GPU).eval()
    preds, gts = evaluate(teacher, test_loader, DEV_GPU)
    record("教师BERT fp32", teacher, DEV_GPU, preds, gts, note="GPU延迟")
    # CPU 延迟单测（不重复算 acc）
    teacher_cpu = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(cats))
    teacher_cpu.load_state_dict(ckpt["state_dict"]); teacher_cpu.eval()
    record("教师BERT fp32-CPU", teacher_cpu, DEV_CPU,
           preds=None, gts=None, note="同参数CPU延迟对照")
    del teacher_cpu

    # ============ 1) INT8 动态量化（DQ，讲义口径：CPU） ============
    try:
        q = torch.ao.quantization.quantize_dynamic(
            teacher.cpu(), {nn.Linear}, dtype=torch.qint8)
        q.eval()
        preds, gts = evaluate(q, test_loader, DEV_CPU)
        record("INT8动态量化(CPU)", q, DEV_CPU, preds, gts,
               note="quantize_dynamic, 讲义DQ路线")
        torch.save(q.state_dict(), "/tmp/int8_state.pt")
        RESULTS["INT8动态量化(CPU)"]["size_mb"] = round(
            os.path.getsize("/tmp/int8_state.pt") / 1048576.0, 1)
        print("[record] INT8 存盘体积校正:",
              RESULTS["INT8动态量化(CPU)"]["size_mb"], flush=True)
    except Exception as e:
        RESULTS["INT8动态量化(CPU)"] = {"error": repr(e)[:300]}
        print("[ERROR] INT8:", traceback.format_exc()[-500:], flush=True)

    # ============ 2) 4bit 量化（bitsandbytes NF4，GPU） ============
    try:
        import bitsandbytes as bnb
        print("bnb", bnb.__version__, flush=True)
        m4 = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(cats))
        state = ckpt["state_dict"]
        # 只换 encoder 里的 Linear 为 4bit（embeddings 与 classifier 保留）
        for name, module in list(m4.named_modules()):
            if isinstance(module, nn.Linear) and ".encoder." in name + ".":
                parent = m4.get_submodule(name.rsplit(".", 1)[0])
                leaf = name.rsplit(".", 1)[1]
                setattr(parent, leaf, bnb.nn.Linear4bit(
                    module.in_features, module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=torch.float16, quant_type="nf4"))
        missing, unexpected = m4.load_state_dict(state, strict=False)
        m4.to(DEV_GPU).eval()                    # 移到 GPU 时 bnb 触发量化
        preds, gts = evaluate(m4, test_loader, DEV_GPU)
        record("NF4 4bit量化(GPU)", m4, DEV_GPU, preds, gts,
               note="bnb NF4, encoder Linear 4bit, embeddings/classifier 保留")
    except Exception as e:
        RESULTS["NF4 4bit量化(GPU)"] = {"error": repr(e)[:300]}
        print("[ERROR] NF4:", traceback.format_exc()[-500:], flush=True)

    # ============ 3) 软标签蒸馏（4 层学生，T=2, α=0.7） ============
    try:
        T, ALPHA, EPOCHS, PER_EPOCH, BATCH = 2, 0.7, 12, 1024, 32
        cfg = BertConfig.from_pretrained(MODEL_PATH, num_labels=len(cats),
                                         num_hidden_layers=4)
        student = BertForSequenceClassification(cfg)
        t_sd = teacher.state_dict()
        init = {}
        for k, v in t_sd.items():
            if k.startswith("bert.encoder.layer."):
                idx = int(k.split(".")[3])
                if idx < 4:
                    init[k.replace("layer.%d." % idx, "layer.%d." % idx, 1)] = v
            elif not k.startswith("bert.encoder.layer."):
                init[k] = v                     # embeddings / pooler / classifier
        miss, unexp = student.load_state_dict(init, strict=False)
        print("学生初始化 missing=%d unexpected=%d" % (len(miss), len(unexp)), flush=True)
        student.to(DEV_GPU)
        teacher_gpu = teacher.to(DEV_GPU).eval()
        optim = torch.optim.AdamW(student.parameters(), lr=2e-5)
        rng = np.random.RandomState(SEED)
        ce = nn.CrossEntropyLoss()
        kl = nn.KLDivLoss(reduction="batchmean")
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
                    t_logits = teacher_gpu(input_ids=ids, attention_mask=att).logits
                s_logits = student(input_ids=ids, attention_mask=att).logits
                hard = ce(s_logits, bl)
                soft = kl(torch.log_softmax(s_logits / T, -1),
                          torch.softmax(t_logits / T, -1)) * (T * T)
                loss = (1 - ALPHA) * hard + ALPHA * soft
                optim.zero_grad(); loss.backward(); optim.step()
            print("  KD epoch %d/%d loss=%.4f (%.0fs)"
                  % (ep, EPOCHS, loss.item(), time.time() - t0), flush=True)
        kd_train_s = round(time.time() - t0, 1)
        student.eval()
        preds, gts = evaluate(student, test_loader, DEV_GPU)
        record("蒸馏4层学生 fp32", student, DEV_GPU, preds, gts,
               note="T=2 a=0.7 12轮x1024 抽样, KD训练%.0fs" % kd_train_s)
        RESULTS["蒸馏4层学生 fp32"]["train_s"] = kd_train_s
        # 3b) 学生再 INT8：部署组合拳
        q4 = torch.ao.quantization.quantize_dynamic(
            student.cpu(), {nn.Linear}, dtype=torch.qint8)
        q4.eval()
        preds, gts = evaluate(q4, test_loader, DEV_CPU)
        record("蒸馏4层学生 INT8(CPU)", q4, DEV_CPU, preds, gts,
               note="学生再量化的部署组合")
    except Exception as e:
        RESULTS["蒸馏4层学生"] = {"error": repr(e)[:300]}
        print("[ERROR] KD:", traceback.format_exc()[-500:], flush=True)

    # ============ 4) 非结构化剪枝（全局 L1，30% / 50%） ============
    try:
        from torch.nn.utils import prune
        for amount in (0.3, 0.5):
            p = BertForSequenceClassification.from_pretrained(
                MODEL_PATH, num_labels=len(cats))
            p.load_state_dict(ckpt["state_dict"])
            params = []
            for name, mod in p.named_modules():
                if isinstance(mod, nn.Linear) and "classifier" not in name:
                    params.append((mod, "weight"))
            prune.global_unstructured(params, pruning_method=prune.L1Unstructured,
                                      amount=amount)
            for mod, _ in params:
                prune.remove(mod, "weight")
            p.to(DEV_GPU).eval()
            preds, gts = evaluate(p, test_loader, DEV_GPU)
            record("剪枝%d%%(非结构化)" % int(amount * 100), p, DEV_GPU, preds, gts,
                   note="global_unstructured L1, 逻辑剪枝(体积不变是预期)")
            # CPU 延迟对照一次（30% 档）
            if amount == 0.3:
                record("剪枝30%%(CPU)", p.cpu(), DEV_CPU)
    except Exception as e:
        RESULTS["剪枝"] = {"error": repr(e)[:300]}
        print("[ERROR] 剪枝:", traceback.format_exc()[-500:], flush=True)

    json.dump({"n_test": N_TEST, "batch": 128, "max_len": MAX_LEN,
               "gpu": torch.cuda.get_device_name(0) if DEV_GPU.type == "cuda" else "cpu",
               "results": RESULTS},
              open("/kaggle/working/bert_compress.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("===RESULT_START===")
    print(json.dumps(RESULTS, ensure_ascii=False))
    print("===RESULT_END===")
    print("done.", flush=True)


if __name__ == "__main__":
    main()
