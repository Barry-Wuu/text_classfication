# -*- coding: utf-8 -*-
"""Kaggle 云端：BERT 续训 —— 从上游内核「最后一轮(epoch10)」的权重继续训练，直到测试准确率明显下降（过拟合）。

与首次训练(compare_bert_finetune.py)的区别只有一个：**起点是已保存的 epoch10 权重**，
而不是重新加载 bert-base-chinese 的原始预训练权重。训练协议完全不变：

  每 epoch 从训练集**随机抽 1024 条**（绝不把整份 23852 条训练集喂进去）-> 训该 epoch
  -> 记录 用时 / 训练loss / 训练acc / 测试acc / 测试macroF1 -> 保存一次权重 bert_epoch{N}.pt

停止条件：test_acc 相对续训期间的历史最佳下降 >= DROP_ACC(0.03) 时判定"明显下降、已过拟合"，
再跑一轮确认后停止；硬上限 MAX_EPOCH=30 兜底。

上游：wubarry/textcls-bert-finetune 的 output（kernel_sources 挂载），其中有
      bert_epoch10.pt（最后一轮权重）与 result.json（前 10 轮曲线）。
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
START_EPOCH = 11          # 第一轮续训的序号（前 10 轮已在上游跑完）
MAX_EPOCH = 30            # 硬上限
SAMPLES_PER_EPOCH = 1024  # 每 epoch 抽样的样本数（协议核心：抽样，不整份训练）
BATCH = 32
LR = 2e-5
DROP_ACC = 0.03           # test_acc 相对最佳下降该幅度即判"明显下降"
MODEL_PATH = "bert-base-chinese"
UPSTREAM_SLUG = "textcls-bert-finetune"

RAW_URL = "https://raw.githubusercontent.com/paixiaoxin66/text_classfication/Barry/train_augmented.csv"
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


def find_upstream():
    """定位上游 output 挂载点，返回 {basename: fullpath}。"""
    roots = ['/kaggle/input/notebooks/wubarry/' + UPSTREAM_SLUG,
             '/kaggle/input/' + UPSTREAM_SLUG]
    for r in roots:
        if os.path.isdir(r):
            found = {}
            for cur, _, fs in os.walk(r):
                for f in fs:
                    found.setdefault(f, os.path.join(cur, f))
            return found, r
    found = {}
    for cur, _, fs in os.walk('/kaggle/input'):
        for f in fs:
            found.setdefault(f, os.path.join(cur, f))
    return found, '/kaggle/input'


def main():
    print("=" * 62, flush=True)
    import platform
    print("  python:", platform.python_version())
    print("  torch:", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)
    print(f"  协议: 每 epoch 随机抽 {SAMPLES_PER_EPOCH} 条(不整份训练集), "
          f"续训 {START_EPOCH}~{MAX_EPOCH}, batch={BATCH}, lr={LR}, max_len={MAX_LEN}", flush=True)
    print(f"  停止: test_acc 相对最佳下降 >= {DROP_ACC}", flush=True)

    found, root = find_upstream()
    print("\n上游挂载点:", root, flush=True)
    for k in sorted(found):
        print("   %-24s %12d B" % (k, os.path.getsize(found[k])), flush=True)

    ckpt_name = f"bert_epoch{START_EPOCH - 1}.pt"
    if ckpt_name not in found:
        raise RuntimeError(f"上游未找到续训起点权重 {ckpt_name}")
    ckpt_path = found[ckpt_name]
    print("\n续训起点:", ckpt_path, os.path.getsize(ckpt_path), "B", flush=True)

    # 上游前 10 轮曲线（若存在）
    prev_log = []
    if "result.json" in found:
        try:
            j = json.load(open(found["result.json"], encoding="utf-8"))
            prev_log = j.get("epoch_log", [])
            print("上游 result.json 已载入，前 %d 轮曲线" % len(prev_log), flush=True)
        except Exception as e:
            print("  result.json 读取失败:", repr(e), flush=True)

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
    yte = y[test]

    global tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(MODEL_PATH)
    test_ds = TxtDs([texts[i] for i in test], yte)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("  device:", device, flush=True)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=len(cats))
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = sd["state_dict"] if isinstance(sd, dict) and "state_dict" in sd else sd
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("  已载入上游权重 | missing=%d unexpected=%d" % (len(missing), len(unexpected)), flush=True)
    if prev_log:
        print("  上游最后一轮记录:", prev_log[-1], flush=True)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    epoch_log = list(prev_log)
    best_acc = prev_log[-1]["test_acc"] if prev_log else 0.0
    if prev_log:
        best_acc = max(r["test_acc"] for r in prev_log)
    best_ep = max([r["epoch"] for r in prev_log], default=0)
    overfit_ep = None
    total_train_dt = 0.0
    t_start = time.time()

    for epoch in range(START_EPOCH, MAX_EPOCH + 1):
        # 每 epoch 从训练集随机抽 1024 条（种子=epoch，与首次训练同规则）
        rng = np.random.default_rng(epoch)
        ep_idx = rng.choice(train, size=SAMPLES_PER_EPOCH, replace=False)
        ep_ds = TxtDs([texts[i] for i in ep_idx], y[ep_idx])
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

        ckpt = os.path.join("/kaggle/working", f"bert_epoch{epoch}.pt")
        torch.save({"state_dict": model.state_dict(), "num_labels": len(cats),
                    "epoch": epoch, "test_macro_f1": round(float(test_f1), 4),
                    "test_acc": round(float(test_acc), 4)}, ckpt)

        drop = best_acc - test_acc
        flag = ""
        if test_acc > best_acc:
            best_acc, best_ep = test_acc, epoch
            flag = "  <- 新最佳"
        rec = dict(epoch=epoch, train_s=round(train_dt, 2), train_loss=round(train_loss, 4),
                   train_acc=round(train_acc, 4), test_acc=round(float(test_acc), 4),
                   test_macro_f1=round(float(test_f1), 4),
                   drop_from_best=round(float(drop), 4))
        epoch_log.append(rec)
        print(f"Epoch {epoch}/{MAX_EPOCH}  用时={train_dt:.1f}s  训练loss={train_loss:.4f}  "
              f"训练acc={train_acc:.4f}  测试acc={test_acc:.4f}  "
              f"测试macroF1={test_f1:.4f}  较最佳={-drop:+.4f}{flag}  -> 已存 {os.path.basename(ckpt)}",
              flush=True)

        if drop >= DROP_ACC:
            overfit_ep = epoch
            print(f"\n>>> 判定过拟合：epoch {epoch} 的 测试acc={test_acc:.4f} 比历史最佳 "
                  f"{best_acc:.4f}(epoch {best_ep}) 低 {drop:.4f} >= {DROP_ACC}，停止续训。", flush=True)
            break

    total_dt = time.time() - t_start
    print(f"\n续训墙钟={total_dt:.1f}s  起点=epoch{START_EPOCH-1}  结束=epoch{epoch_log[-1]['epoch']}  "
          f"全程最佳 test_acc={best_acc:.4f}(epoch {best_ep})", flush=True)
    if overfit_ep:
        print(f"过拟合拐点: epoch {overfit_ep}", flush=True)

    best_rec = next((r for r in epoch_log if r["epoch"] == best_ep), epoch_log[-1])
    row = dict(
        name="BERT微调(续训至过拟合)",
        macro_f1=round(float(best_rec["test_macro_f1"]), 4),
        acc=round(float(best_rec["test_acc"]), 4),
        fit_s=round(total_train_dt, 1),
        infer_s=0.0,
        extra=f"(从epoch{START_EPOCH-1}续训, best=epoch{best_ep}, overfit@{overfit_ep}, "
              f"抽样{SAMPLES_PER_EPOCH}/epoch, max_len={MAX_LEN})",
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
    print("已写 /kaggle/working/result.json", flush=True)


if __name__ == "__main__":
    main()
