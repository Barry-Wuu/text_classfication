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

# 语气词白名单：仅压缩这些字的重复（啊啊啊→啊）
# 刻意不含 充/咚/多/团/滋/刚/都/达 等 —— 那些是品牌名或拟声词（拼多多、咚咚咚），压缩会破坏语义
_TONE_WORDS = "啊呀吧呢吗嘛哦噢喔诶唉哎嗨哈嘿呵嘻嗯唔哟咦哇咯啦嘞唷"

# emoji 与"非汉字型情绪符号"（图片占位符、表情、键帽等）的匹配
# 注意：不含中文字、不含 ASCII 数字字母，避免误伤
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # 各类 emoji、麻将、扑克、补充符号
    "\U0001F1E6-\U0001F1FF"   # 区域指示符（国旗）
    "\U00002600-\U000027BF"   # 杂项符号与装饰符号（☀✅❌✖★♪ 等）
    "\U00002190-\U000021FF"   # 箭头（→←↑↓）
    "\U00002B00-\U00002BFF"   # 杂项符号与箭头
    "\U0000FE00-\U0000FE0F"   # 变体选择符（emoji 修饰，如 ✖️）
    "\U0000200D"              # 零宽连接符（组合 emoji）
    "\U000020E3"              # 组合包围键帽（1️⃣ 的组成）
    "\U0000FFFC"              # 对象替换符（图片占位）
    "\U000024C2"              # 圈 M
    "\U0000FFFD"              # 替换字符（乱码）
    "]+"
)


def fix_rlo(text: str) -> str:
    """修复 U+202E 方向控制符：将 RLO…PDF 区间内容反转，再清落单控制符"""
    text = re.sub(r"\u202e(.*?)\u202c", lambda m: m.group(1)[::-1], text)
    return text.replace("\u202e", "").replace("\u202c", "")


def preprocess(text: str, use_opencc: bool = False) -> str:
    # 1. 修复方向控制符污染（必须先做）
    text = fix_rlo(text)

    # 2. 清零宽字符
    text = re.sub(r"[\u200c\u200b\ufeff]", "", text)

    # 3. 剔除 emoji 与非汉字型情绪符号（图片占位、表情、键帽、箭头等）
    #    放在最前，避免它们干扰后续标点/重复处理
    text = _EMOJI_RE.sub("", text)

    # 4. 换行 → 句号（保留断句）
    text = re.sub(r"[\n\r\u2028]+", "。", text)

    # 5. 省略号归一化：连续点 → 单省略号 …（必须早于半角转全角）
    text = re.sub(r"\.{2,}", "…", text)
    text = re.sub(r"…{2,}", "…", text)   # 已有 … 的也压成一个

    # 6. 半角标点转全角（放在标点压缩之前，否则半角 ! ? , 会漏压）
    text = text.translate(_HALF2FULL)

    # 7. 连续标点压缩（！！！→！  。。。→。）
    text = re.sub(r"([，。！？；：、…])\1{1,}", r"\1", text)

    # 8. 重复语气词压缩（啊啊啊→啊），仅限语气词白名单
    text = re.sub(
        r"([" + _TONE_WORDS + r"])\1{1,}",
        r"\1", text)

    # 9. 连续星号压缩（保留脱敏信号）
    text = re.sub(r"\*{2,}", "*", text)

    # 10. 压缩连续空格
    text = re.sub(r" {2,}", " ", text).strip()

    # 11.（可选）繁转简
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
        "含emoji/非汉字符号": sum(1 for r in rows if _EMOJI_RE.search(r["text"])),
        "含换行": sum(1 for r in rows if "\n" in r["text"] or "\r" in r["text"]),
        "含省略号(>=2点)": sum(1 for r in rows if re.search(r"\.{2,}|…{2,}", r["text"])),
        "含重复标点": sum(1 for r in rows if re.search(r"([，。！？；：、…])\1", r["text"])),
        "含重复语气词": sum(1 for r in rows if re.search(r"([" + _TONE_WORDS + r"])\1", r["text"])),
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
