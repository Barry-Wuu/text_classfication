# -*- coding: utf-8 -*-
"""模型压缩四方案的对比图（一张三面板）。

输入：resources/bert_compress.json（Kaggle 内核 wubarry/textcls-bert-compress 产出）
输出：resources/figures/19_compress_compare.png

面板：A 宏F1（柱上标 acc）/ B 模型体积 MB / C 推理延迟 ms/样本（注明设备）。
注意各方案设备不同（量化 DQ 只能 CPU、NF4 只能 GPU），延迟只能在同设备内
比较——每个面板内按设备分组，跨设备比较没有意义，这是诚实口径。
"""
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False

DPI = 140
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "resources", "figures")
os.makedirs(OUT, exist_ok=True)

d = json.load(open(os.path.join(ROOT, "resources", "bert_compress.json"),
                   encoding="utf-8"))
R = d["results"]


def get(name, key):
    v = R.get(name, {})
    return v.get(key)


# 展示顺序与设备标注（JSON 键, 图中显示名, 设备, 颜色）
ITEMS = [
    ("教师BERT fp32",              "原版 BERT（教师）fp32",   "GPU", "#d946ef"),
    ("NF4 4bit量化(GPU)",          "NF4 4bit 量化",           "GPU", "#3b82f6"),
    ("剪枝30%(非结构化)",           "非结构化剪枝 30%",         "GPU", "#f59e0b"),
    ("剪枝50%(非结构化)",           "非结构化剪枝 50%",         "GPU", "#f59e0b"),
    ("蒸馏LSTM学生 256维(20轮)",    "蒸馏 LSTM 学生 256 维（20 轮）", "GPU", "#10b981"),
    ("蒸馏LSTM学生 128维(8轮)",     "蒸馏 LSTM 学生 128 维（8 轮）",  "GPU", "#10b981"),
    ("教师BERT fp32-CPU",          "原版 BERT（教师）fp32",   "CPU", "#d946ef"),
    ("INT8动态量化(CPU)",          "INT8 动态量化（DQ）",      "CPU", "#3b82f6"),
    ("剪枝30%(CPU)",               "非结构化剪枝 30%",         "CPU", "#f59e0b"),
    ("蒸馏LSTM学生 256维(20轮)-CPU", "蒸馏 LSTM 学生 256 维",    "CPU", "#10b981"),
    ("蒸馏LSTM学生 256维+INT8(CPU)", "蒸馏 LSTM 学生 256 维 + INT8", "CPU", "#10b981"),
    ("蒸馏LSTM学生 128维(8轮)-CPU",  "蒸馏 LSTM 学生 128 维",    "CPU", "#10b981"),
    ("蒸馏LSTM学生 128维+INT8(CPU)", "蒸馏 LSTM 学生 128 维 + INT8", "CPU", "#10b981"),
]
items = [(k, lab, dev, c) for k, lab, dev, c in ITEMS
         if get(k, "macro_f1") is not None
         or (get(k, "ms_per_sample") is not None and get(k, "size_mb") is not None)]
print("可用方案:", [lab for _, lab, _, _ in items])

fig, axes = plt.subplots(1, 3, figsize=(15.2, 6.4), facecolor="white", dpi=DPI)

labels = [f"{lab}\n[{dev}]" for _, lab, dev, _ in items]
colors = [c for _, _, _, c in items]
ys = list(range(len(items)))[::-1]

def num(v):
    """缺测的指标记为 0（柱不出），标注用 - 占位。"""
    return 0.0 if v is None else v


# ---- A: 精度 ----
ax = axes[0]
f1s = [num(get(k, "macro_f1")) for k, _, _, _ in items]
accs = [get(k, "acc") for k, _, _, _ in items]
ax.barh(ys, f1s, color=colors, alpha=0.85, height=0.62)
lo = min(v for v in f1s if v) - 0.04
ax.set_xlim(lo, max(f1s) + 0.015)
for y, f1, acc in zip(ys, f1s, accs):
    txt = "未测(纯延迟对照)" if f1 == 0 else f"F1 {f1:.4f}\nacc {acc:.4f}"
    ax.text((f1 if f1 else lo) + 0.002, y, txt, va="center",
            fontsize=8.4, color="#0f172a")
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=8.6)
ax.set_title("精度（越高越好）", fontsize=12, fontweight="bold", color="#0f172a")
ax.set_xlabel("macro-F1", fontsize=10)
ax.grid(axis="x", alpha=0.25)

# ---- B: 体积 ----
ax = axes[1]
sizes = [num(get(k, "size_mb")) for k, _, _, _ in items]
ax.barh(ys, sizes, color=colors, alpha=0.85, height=0.62)
for y, s in zip(ys, sizes):
    ax.text(s + max(sizes) * 0.012, y, f"{s:.1f} MB" if s else "-", va="center",
            fontsize=9, color="#0f172a")
ax.set_xlim(0, max(sizes) * 1.22)
ax.set_yticks(ys); ax.set_yticklabels([]) 
ax.set_title("模型体积（越小越好）", fontsize=12, fontweight="bold", color="#0f172a")
ax.set_xlabel("state_dict 存盘 MB", fontsize=10)
ax.grid(axis="x", alpha=0.25)

# ---- C: 延迟 ----
ax = axes[2]
lats = [num(get(k, "ms_per_sample")) for k, _, _, _ in items]
ax.barh(ys, lats, color=colors, alpha=0.85, height=0.62)
for y, l in zip(ys, lats):
    ax.text(l + max(lats) * 0.012, y, f"{l:.2f} ms" if l else "-", va="center",
            fontsize=9, color="#0f172a")
ax.set_xlim(0, max(lats) * 1.22)
ax.set_yticks(ys); ax.set_yticklabels([])
ax.set_title("推理延迟（越低越好 · 同设备内比较）", fontsize=12,
             fontweight="bold", color="#0f172a")
ax.set_xlabel("ms / 样本（batch=128, max_len=128）", fontsize=10)
ax.grid(axis="x", alpha=0.25)

fig.suptitle("BERT 打榜模型（epoch 59）四种压缩方案对比 · 无泄漏测试集 1,378 条 · "
             "量化省精度 / 蒸馏省规模 / 剪枝省参数",
             fontsize=13.5, fontweight="bold", color="#0f172a", y=0.99)
fig.tight_layout(rect=(0, 0, 1, 0.93))
p_out = os.path.join(OUT, "19_compress_compare.png")
fig.savefig(p_out, bbox_inches="tight", facecolor="white")
print("saved", p_out)
