# -*- coding: utf-8 -*-
"""合并各压缩内核输出为 resources/bert_compress.json。

- int8 / nf4b / prune 三个内核结果直接取用；
- KD 有两版：kd2（128 维学生 8 轮）与 kd3（256 维学生 20 轮），改名为可区分的
  名称后合并；kd（初版训练不足）只作反面证据写进 README，不进本表。
"""
import os
import json

BASE = r"D:\B++\WorkBuddy\kaggle_bert_compress"
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "resources")

KD2_RENAME = {
    "蒸馏LSTM学生 fp32": "蒸馏LSTM学生 128维(8轮)",
    "蒸馏LSTM学生 fp32-CPU": "蒸馏LSTM学生 128维(8轮)-CPU",
    "蒸馏LSTM学生 INT8(CPU)": "蒸馏LSTM学生 128维+INT8(CPU)",
}
KD3_RENAME = {
    "蒸馏LSTM学生 fp32": "蒸馏LSTM学生 256维(20轮)",
    "蒸馏LSTM学生 fp32-CPU": "蒸馏LSTM学生 256维(20轮)-CPU",
    "蒸馏LSTM学生 INT8(CPU)": "蒸馏LSTM学生 256维+INT8(CPU)",
}
SOURCES = [
    ("int8", "out_int8", {}),
    ("nf4", "out_nf4b", {}),
    ("prune", "out_prune", {"剪枝30%%(CPU)": "剪枝30%(CPU)"}),
    ("kd2", "out_kd2", KD2_RENAME),
    ("kd3", "out_kd3", KD3_RENAME),
]

merged = {"results": {}, "sources": {}, "meta": {}}
for tag, d, rename in SOURCES:
    d = os.path.join(BASE, d)
    if not os.path.isdir(d):
        print("missing", d)
        continue
    for f in os.listdir(d):
        if f.startswith("bert_cmp") and f.endswith(".json"):
            j = json.load(open(os.path.join(d, f), encoding="utf-8"))
            for k in ("n_test", "batch", "max_len"):
                if j.get(k) is not None:
                    merged["meta"][k] = j[k]
            for k, v in j.get("results", {}).items():
                name = rename.get(k, k)
                merged["results"][name] = v
                merged["sources"][name] = "wubarry/textcls-cmp-" + tag
out = os.path.join(RES, "bert_compress.json")
json.dump(merged, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("merged ->", out)
for k, v in merged["results"].items():
    print(f"{k:30s} {json.dumps(v, ensure_ascii=False)}")
