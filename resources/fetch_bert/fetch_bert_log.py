# -*- coding: utf-8 -*-
"""挂载上游 textcls-bert-finetune 的 output，打印运行日志/错误/结果（不下载大权重）。"""
import os, sys, json

BASE = "/kaggle/input/notebooks/wubarry/textcls-bert-finetune"
print("挂载点:", BASE, flush=True)
found = []
for root, dirs, files in os.walk(BASE):
    for f in files:
        p = os.path.join(root, f)
        try:
            sz = os.path.getsize(p)
        except Exception:
            sz = -1
        found.append((f, sz, p))
        print(f"  FILE {f}  {sz} bytes  -> {p}", flush=True)

for name in ("run.log", "FATAL.txt", "result.json"):
    for f, sz, p in found:
        if f == name and 0 <= sz < 8_000_000:
            print(f"\n========== {name} ==========", flush=True)
            try:
                print(open(p, encoding="utf-8", errors="replace").read()[-9000:], flush=True)
            except Exception as e:
                print("读取失败:", repr(e), flush=True)
