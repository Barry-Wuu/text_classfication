# -*- coding: utf-8 -*-
"""
消费者投诉文本分类数据集 —— 文本预处理

用法:
    python eda/preprocess.py            # 只统计，不落盘
    python eda/preprocess.py --write    # 清洗并输出 clean.csv（写入项目根目录）

处理项（顺序不可调换，原因见同目录 PREPROCESS.md 第八节）:
    1. 修复 U+202E 方向控制符污染（字序错乱）
    2. 清零宽字符
    3. 换行 → 句号（保留断句语义）
    4. 省略号归一化（必须早于半角转全角）
    5. 连续标点压缩
    6. 连续星号压缩（保留脱敏信号）
    7. 半角标点转全角
    8. 压缩连续空格
    9.（可选）繁转简

仅依赖标准库；繁转简为可选（需 opencc）。
"""
import os
import re
import csv
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(ROOT, "train_augmented.csv")
OUT_PATH = os.path.join(ROOT, "clean.csv")

# 半角 → 全角 标点映射（仅这些确定是中文标点用法的才转）
# 故意不含 - / ' ，因为它们在本数据集中几乎全是数字化/型号化用途（见 PREPROCESS.md 第四节）
_HALF2FULL = str.maketrans({
    ",": "，", ":": "：", ";": "；", "!": "！", "?": "？",
    "(": "（", ")": "）", '"': "”",
})


def fix_rlo(text: str) -> str:
    """修复 U+202E 方向控制符：将 RLO…PDF 区间内容反转，再清落单控制符"""
    text = re.sub(r"\u202e(.*?)\u202c", lambda m: m.group(1)[::-1], text)
    return text.replace("\u202e", "").replace("\u202c", "")


def preprocess(text: str, use_opencc: bool = False) -> str:
    # 1. 修复方向控制符污染（必须先做）
    text = fix_rlo(text)

    # 2. 清零宽字符
    text = re.sub(r"[\u200c\u200b\ufeff]", "", text)

    # 3. 换行 → 句号（保留断句）
    text = re.sub(r"[\n\r\u2028]+", "。", text)

    # 4. 省略号归一化（必须早于半角转全角）
    text = re.sub(r"\.{2,}", "……", text)

    # 5. 半角标点转全角（放在标点压缩之前，否则半角 ! ? , 会漏压）
    text = text.translate(_HALF2FULL)

    # 6. 连续标点压缩（此时半角已统一为全角）
    text = re.sub(r"([，。！？；：、])\1{1,}", r"\1", text)

    # 7. 连续星号压缩（保留脱敏信号）
    text = re.sub(r"\*{2,}", "*", text)

    # 8. 压缩连续空格
    text = re.sub(r" {2,}", " ", text).strip()

    # 9.（可选）繁转简
    if use_opencc:
        from opencc import OpenCC
        text = OpenCC("t2s").convert(text)

    return text


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for r in reader:
            rows.append(r)
    return rows, fields


def scan(rows):
    """统计需要处理的项"""
    stat = {
        "含RLO控制符": sum(1 for r in rows if "\u202e" in r["text"] or "\u202c" in r["text"]),
        "含零宽字符": sum(1 for r in rows if re.search(r"[\u200c\u200b\ufeff\u2028]", r["text"])),
        "含换行": sum(1 for r in rows if "\n" in r["text"] or "\r" in r["text"]),
        "含省略号(>=2点)": sum(1 for r in rows if re.search(r"\.{2,}", r["text"])),
        "含重复标点": sum(1 for r in rows if re.search(r"([，。！？；：、])\1", r["text"])),
        "含连续星号": sum(1 for r in rows if re.search(r"\*{2,}", r["text"])),
        "含半角标点": sum(1 for r in rows if re.search(r'[":;!?()]', r["text"])),
    }
    return stat


def main():
    write = "--write" in sys.argv

    print(f"读取: {CSV_PATH}")
    rows, fields = load_rows(CSV_PATH)
    print(f"样本数: {len(rows)}\n")

    before = scan(rows)
    print("【处理前统计】")
    for k, v in before.items():
        print(f"  {k}: {v}")

    # 逐条处理
    changed = 0
    examples = []
    for r in rows:
        old = r["text"]
        new = preprocess(old, use_opencc=False)
        if new != old:
            changed += 1
            if len(examples) < 5 and "\u202e" in old:
                examples.append((old, new))
        r["text"] = new

    print(f"\n【处理结果】")
    print(f"  发生变化的样本: {changed} ({changed / len(rows) * 100:.2f}%)")

    after = scan(rows)
    print("\n【处理后统计】")
    for k, v in after.items():
        print(f"  {k}: {v}")

    if examples:
        print("\n【RLO 修复对照样例】")
        for old, new in examples:
            print(f"  原: {old[:80]}")
            print(f"  修: {new[:80]}\n")

    if write:
        with open(OUT_PATH, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"已写出: {OUT_PATH}")
    else:
        print("\n（未落盘；加 --write 生成 clean.csv）")


if __name__ == "__main__":
    main()
