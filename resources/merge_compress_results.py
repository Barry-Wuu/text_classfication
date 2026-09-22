# -*- coding: utf-8 -*-
"""合并四个压缩内核的输出为 resources/bert_compress.json。

输入：D:\B++\WorkBuddy\kaggle_bert_compress\out_{int8,nf4,prune,kd}\bert_cmp_*.json
输出：resources/bert_compress.json（results 扁平合并，冲突以后到的内核为准）
"""
import os
import json

BASE = r"D:\B++\WorkBuddy\kaggle_bert_compress"
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "resources")

merged = {"results": {}, "sources": {}}
for tag in ("int8", "nf4", "prune", "kd"):
    d = os.path.join(BASE, "out_" + tag)
    if not os.path.isdir(d):
        print("missing", d)
        continue
    for f in os.listdir(d):
        if f.startswith("bert_cmp") and f.endswith(".json"):
            j = json.load(open(os.path.join(d, f), encoding="utf-8"))
            merged["n_test"] = j.get("n_test")
            merged["batch"] = j.get("batch")
            merged["max_len"] = j.get("max_len")
            merged["gpu"] = j.get("gpu")
            for k, v in j.get("results", {}).items():
                merged["results"][k] = v
                merged["sources"][k] = "wubarry/textcls-cmp-" + tag
merged["results"] = dict(sorted(merged["results"].items()))
out = os.path.join(RES, "bert_compress.json")
json.dump(merged, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("merged ->", out)
for k, v in merged["results"].items():
    print(f"{k:28s} {json.dumps(v, ensure_ascii=False)}")
