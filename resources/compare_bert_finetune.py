# -*- coding: utf-8 -*-
"""Kaggle 云端：BERT（bert-base-chinese）正经微调 —— 参照用户笔记本的"每 epoch 随机抽 1024"协议。

与之前 compare_tensors.py 的"误用"（冻结 BERT 抽 [CLS] 向量 + LinearSVC，权重零更新）不同，
本脚本对 BertForSequenceClassification 做**端到端微调**（全参数可训），并严格按用户指定的协议：

  每 epoch 从训练集随机抽 1024 条 -> 训练该 epoch -> 观察 用时 / 训练集准确率 / 测试集准确率
  -> 保存一次权重；共训练 10 个 epoch（每 epoch 的随机种子 = epoch 序号，故前 3 轮与 3-epoch 版一致，
  相当于从第 3 轮继续训到第 10 轮）。

数据：从 GitHub 拉 train_augmented.csv，复用 compare_no_leak.py 的无泄漏分组划分
      （测试=1378 全原始，源 id 不进训练），保证测试集与 16 方法对比表完全一致、可直接比 macro-F1。
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

import os as _os
_os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertForSequenceClassification

SEED = 42
MAX_LEN = 128
EPOCHS = 10
SAMPLES_PER_EPOCH = 1024
BATCH = 32
LR = 2e-5
MODEL_PATH = "bert-base-chinese"

RAW_URL = "https://raw.githubusercontent.com/Barry-Wuu/text_classfication/main/train_augmented.csv"
MIRROR_URL = "https://ghfast.top/" + RAW_URL


def fetch():
    import requests
    dst = '/kaggle/working/train_augmented.csv'
    for url in (RAW_URL, MIRROR_URL):
        try:
            print("  下载:", url, flush=True)
            r = requests.get(url, timeout=240)
            print("    HTTP", r.status_code, "大小", len(r.content), flush=True)
            if r.status_code == 200 and len(r.content) > 1_000_000:
                open(dst, 'wb').write(r.content)
                return dst
        except Exception as e:
            print("    失败:", repr(e), flush=True)
    raise RuntimeError("train_augmented.csv 下载失败")


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
    train = np.concatenate([tr_ori, train_aug_idx])
    test = te_ori
    return train, test, test_src


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
    print("=" * 62, flush=True)
    import platform
    print("  python:", platform.python_version())
    print("  torch:", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)
    print(f"  协议: 每 epoch 随机抽 {SAMPLES_PER_EPOCH} 条, 训练 {EPOCHS} epoch, "
          f"batch={BATCH}, lr={LR}, max_len={MAX_LEN}", flush=True)

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
          f" / 测试 {len(test)}（全原始, 源 {len(test_src)} 个）", flush=True)
    ytr_all, yte = y[train], y[test]

    global tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    test_ds = TxtDs([texts[i] for i in test], yte)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("  device:", device, flush=True)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(cats))
    model.to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  可训练参数(全量微调): {n_train:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    epoch_log = []
    best_f1, best_ep = 0.0, 0
    total_train_dt = 0.0
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        # 每 epoch 从训练集随机抽 1024 条
        rng = np.random.default_rng(epoch)
        ep_idx = rng.choice(train, size=SAMPLES_PER_EPOCH, replace=False)
        ep_texts = [texts[i] for i in ep_idx]
        ep_labels = y[ep_idx]
        ep_ds = TxtDs(ep_texts, ep_labels)
        ep_loader = DataLoader(ep_ds, batch_size=BATCH, shuffle=True)

        model.train()
        t0 = time.time()
        running_loss, total, correct = 0.0, 0, 0
        for batch in ep_loader:
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            correct += (out.logits.argmax(-1) == labels).sum().item()
        train_dt = time.time() - t0
        total_train_dt += train_dt
        train_acc = correct / total
        train_loss = running_loss / total

        # 评估测试集（与对比表同 1378 条）
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attn = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                logits = model(input_ids=input_ids, attention_mask=attn).logits
                all_preds.extend(logits.argmax(-1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        test_acc = accuracy_score(all_labels, all_preds)
        test_f1 = f1_score(all_labels, all_preds, average="macro")

        # 每 epoch 保存一次权重
        ckpt = os.path.join("/kaggle/working", f"bert_epoch{epoch}.pt")
        torch.save({"state_dict": model.state_dict(), "num_labels": len(cats),
                    "epoch": epoch, "test_macro_f1": round(float(test_f1), 4)},
                   ckpt)

        if test_f1 > best_f1:
            best_f1, best_ep = test_f1, epoch
            torch.save({"state_dict": model.state_dict(), "num_labels": len(cats)},
                       "/kaggle/working/bert_best.pt")

        rec = dict(epoch=epoch, train_s=round(train_dt, 2), train_loss=round(train_loss, 4),
                   train_acc=round(train_acc, 4), test_acc=round(float(test_acc), 4),
                   test_macro_f1=round(float(test_f1), 4))
        epoch_log.append(rec)
        print(f"Epoch {epoch}/{EPOCHS}  用时={train_dt:.1f}s  训练loss={train_loss:.4f}  "
              f"训练acc={train_acc:.4f}  测试acc={test_acc:.4f}  "
              f"测试macroF1={test_f1:.4f}  -> 已存 {os.path.basename(ckpt)}", flush=True)

    total_dt = time.time() - t_start
    print(f"\n总计训练墙钟={total_dt:.1f}s  最佳 epoch={best_ep} (macroF1={best_f1:.4f})", flush=True)

    best_rec = epoch_log[best_ep - 1]
    row = dict(
        name="BERT微调(随机1024×10epoch)",
        macro_f1=round(float(best_f1), 4),
        acc=round(float(best_rec["test_acc"]), 4),
        fit_s=round(total_train_dt, 1),
        infer_s=0.0,
        extra=f"(端到端微调, best of {EPOCHS} epoch, max_len={MAX_LEN}, {device})",
    )
    print("\n===RESULT_START===")
    print(json.dumps({"row": row, "epoch_log": epoch_log,
                      "split": {"train": int(len(train)), "test": int(len(test)),
                                "test_sources": int(len(test_src))}},
                     ensure_ascii=False))
    print("===RESULT_END===")
    json.dump({"row": row, "epoch_log": epoch_log},
              open("/kaggle/working/result.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("已写 /kaggle/working/result.json 与 3 个 epoch 权重 + bert_best.pt", flush=True)


if __name__ == "__main__":
    main()
