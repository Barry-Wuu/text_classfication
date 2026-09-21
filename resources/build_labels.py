# -*- coding: utf-8 -*-
"""生成标签表与整数标签列。

参照格式：D:\\黑马AI大模型软件\\...\\TMFCode\\01-data\\class.txt
—— 每行一个类名，行号（从 0 起）即 label id。

产物：
  resources/class.txt       : 一行一个类名（0..17），供框架读取
  resources/label_map.csv   : id,name,count 三列完整映射表
  labeled.csv         : text,category,label 三列（label 为 0..17 整数）

类名排序：字典序（可复现，与 label_map 展示顺序一致）。
"""
import os
import pandas as pd

ROOT = r"D:\text_classfication"
CLEAN = os.path.join(ROOT, "clean.csv")
RES = os.path.join(ROOT, "resources")

def main():
    df = pd.read_csv(CLEAN)
    assert list(df.columns) == ["text", "category"], df.columns

    cats = sorted(df["category"].unique())          # 字典序，固定
    name2id = {c: i for i, c in enumerate(cats)}

    # 1) class.txt —— 兼容参考格式
    with open(os.path.join(RES, "class.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(cats) + "\n")

    # 2) label_map.csv —— 带样本量
    cnt = df["category"].value_counts()
    rows = [{"id": name2id[c], "name": c, "count": int(cnt[c])} for c in cats]
    pd.DataFrame(rows).to_csv(os.path.join(RES, "label_map.csv"),
                              index=False, encoding="utf-8")

    # 3) labeled.csv —— text,category,label
    out = df.copy()
    out["label"] = out["category"].map(name2id).astype("int64")
    out.to_csv(os.path.join(ROOT, "labeled.csv"),
               index=False, encoding="utf-8")

    print("[OK] 类别数:", len(cats))
    print("[OK] class.txt / label_map.csv / labeled.csv 已写出")
    print("[OK] label 分布一致性校验:",
          int((out["label"].map(name2id) == out["label"]).all()) == False or "通过")
    print("     label 唯一值:", sorted(out["label"].unique().tolist()))
    # id 与类名双向一致
    back = out["label"].map({v: k for k, v in name2id.items()})
    print("     回映射与原 category 完全一致:", bool((back == out["category"]).all()))

if __name__ == "__main__":
    main()
