# -*- coding: utf-8 -*-
"""
消费者投诉文本分类数据集 —— 文本预处理

用法:
    python eda/preprocess.py            # 只统计，不落盘
    python eda/preprocess.py --write    # 清洗并输出 clean.csv（写入项目根目录）

处理项（顺序不可调换，原因见同目录 PREPROCESS.md 第八节）:
    1. 修复 U+202E 方向控制符污染（字序错乱）
    2. 清零宽字符
    3. 剔除 emoji / 非汉字型情绪符号（含颜文字碎片）
    4. 去除无信息量包装括号 【】〖〗[]{}（保留其内内容）
    5. 剔除误用撇号类符号 ′″〞＇'
    6. 圈号数字归一 ①→1
    7. 异体/全角变体符号归一（U+2011→-、×→x、「」→“”、￥→¥ 等）
    8. 罗马数字归一 Ⅰ→1 Ⅱ→2
    9. 全角拉丁字母转半角（Ｍｚ→Mz）
   10. 换行 → 句号（保留断句语义）
   11. 省略号归一化（必须早于半角转全角）
   12. 半角标点转全角
   13. 连续标点压缩
   14. 重复语气词压缩（白名单）
   15. 连续星号压缩（保留脱敏信号）
   16. 压缩连续空格
   17.（可选）繁转简

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

# 异体/全角变体符号 → 常用形态归一（保持语义，只统一写法）
# 注意：- 系列（U+2011 不换行连字符、U+2212 减号、U+FE63 小连字符、U+2013 en dash）统一为普通 ASCII '-'
_SYMBOL_NORM = str.maketrans({
    "\u2011": "-",     # NON-BREAKING HYPHEN 不换行连字符 → -
    "\u2212": "-",     # MINUS SIGN 减号 → -
    "\ufe63": "-",     # SMALL HYPHEN-MINUS 小连字符 ﹣ → -
    "\u2013": "-",     # EN DASH – → -
    "\u2014": "-",     # EM DASH — → -（数据集里多为破折号，统一为连字符）
    "\uff0b": "+",     # 全角加号 ＋ → +
    "\uff0a": "*",     # 全角星号 ＊ → *
    "\uff5e": "~",     # 全角波浪 ～ → ~
    "\uff0e": ".",     # 全角句点 ． → .
    "\uff5c": "|",     # 全角竖线 ｜ → |
    "\u00d7": "x",     # 乘号 × → x（多用于"尺寸 3×4"）
    "\u300c": "“",     # 「 → “
    "\u300d": "”",     # 」 → ”
    "\u3014": "（",    # 〔 → （
    "\u3015": "）",    # 〕 → ）
    "\uff3b": "（",    # ［ → （
    "\uff3d": "）",    # ］ → ）
    "\uffe5": "\u00a5",  # 全角人民币 ￥ → 半角 ¥（统一货币符号）
    "\uff03": "#",     # 全角井号 ＃ → #
    "\u2236": ":",     # RATIO ∶ → :
    "\u2022": "·",     # BULLET • → ·
    "\u25e6": "·",     # WHITE BULLET ◦ → ·
    "\u25cf": "●",     # BLACK CIRCLE 保持（有强调语义）
})

# 误用/多余的撇号类符号（′ ″ 〞 ＇ '）→ 直接剔除
# 说明：′″〞 在数据集中均为误敲（"范围′内""说一声″。"），转成引号反而制造孤立引号，故删除
_STRIP_PRIME_RE = re.compile(r"[\u2032\u2033\u301e\uff07']")

# 罗马数字 → 阿拉伯数字（保留序号/型号语义：Ⅰ型 → 1型，Z7Ⅱ → Z72）
_ROMAN = str.maketrans({
    "Ⅰ": "1", "Ⅱ": "2", "Ⅲ": "3", "Ⅳ": "4", "Ⅴ": "5",
    "Ⅵ": "6", "Ⅶ": "7", "Ⅷ": "8", "Ⅸ": "9", "Ⅹ": "10",
})

# 全角拉丁字母 → 半角（Ｍｚ330 → Mz330）
_FULLWIDTH_ALNUM = str.maketrans(
    {chr(0xFF01 + i): chr(0x21 + i) for i in range(94) if 0xFF21 <= 0xFF01 + i <= 0xFF3A or 0xFF41 <= 0xFF01 + i <= 0xFF5A}
)

# 需剔除的成对/落单包装符号（无信息量，含内容保留、括号去掉）
# 说明：[] 在本数据集只出现在 [ORDER_ID] 这类隐私占位符；{} 仅见孤立笔误（牢骚{行程）
_STRIP_BRACKETS_RE = re.compile(r"[【】〖〗\[\]\{\}]")

# 圈号数字 ①→1 ②→2 …（保留其"序号/数量"语义）
_CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

# 语气词白名单：仅压缩这些字的重复（啊啊啊→啊）
# 刻意不含 充/咚/多/团/滋/刚/都/达 等 —— 那些是品牌名或拟声词（拼多多、咚咚咚），压缩会破坏语义
_TONE_WORDS = "啊呀吧呢吗嘛哦噢喔诶唉哎嗨哈嘿呵嘻嗯唔哟咦哇咯啦嘞唷"

# emoji 与"非汉字型情绪符号"（图片占位符、表情、键帽、颜文字碎片等）的匹配
# 注意：不含中文字、不含常用 ASCII 数字字母与全角标点，避免误伤
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
    "\u30fd\u30fe"            # 日文叠字符 ヽヾ（颜文字碎片 ヽ(´Д`)ﾉ）
    "\uff89\uff9e\uff9d"      # 半角片假名 ﾉﾞﾝ（颜文字碎片）
    "\uff40"                  # 半角反引号 ｀（颜文字碎片）
    "\u00b4\u0060"            # 尖音符 ´ / 反引号 `（"错误的解释﹣﹣"无此，实为颜文字/误用）
    "\u03b1\u0414"            # 希腊 α / 西里尔 Д（QQα音乐、´Д` 噪声字母）
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

    # 4. 去除无信息量的包装括号 【】〖〗[]{}（保留括号内内容）
    text = _STRIP_BRACKETS_RE.sub("", text)

    # 4b. 剔除误用的撇号类符号 ′ ″ 〞 ＇ '（无配对、无合法语义）
    text = _STRIP_PRIME_RE.sub("", text)

    # 5. 圈号数字归一 ①→1
    text = text.translate(str.maketrans({c: str(i + 1) for i, c in enumerate(_CIRCLED)}))

    # 6. 异体/全角变体符号归一（U+2011→-、×→x、「」→“” 等）
    text = text.translate(_SYMBOL_NORM)

    # 6b. 罗马数字归一（Ⅰ型 → 1型，Z7Ⅱ → Z72）
    text = text.translate(_ROMAN)

    # 6c. 全角拉丁字母 → 半角（Ｍｚ330 → Mz330）
    text = text.translate(_FULLWIDTH_ALNUM)

    # 7. 换行 → 句号（保留断句）；若换行前已是标点则不重复补
    text = re.sub(r"(?<=[，。！？；：、…])\s*[\n\r\u2028]+", "", text)
    text = re.sub(r"[\n\r\u2028]+", "。", text)

    # 8. 省略号归一化：连续点 → 单省略号 …（必须早于半角转全角）
    text = re.sub(r"\.{2,}", "…", text)
    text = re.sub(r"…{2,}", "…", text)   # 已有的 … 的也压成一个

    # 9. 半角标点转全角（放在标点压缩之前，否则半角 ! ? , 会漏压）
    text = text.translate(_HALF2FULL)

    # 10. 连续标点压缩（！！！→！  。。。→。）
    text = re.sub(r"([，。！？；：、…])\1{1,}", r"\1", text)

    # 11. 重复语气词压缩（啊啊啊→啊），仅限语气词白名单
    text = re.sub(
        r"([" + _TONE_WORDS + r"])\1{1,}",
        r"\1", text)

    # 12. 连续星号压缩（保留脱敏信号）
    text = re.sub(r"\*{2,}", "*", text)

    # 13. 压缩连续空格
    text = re.sub(r" {2,}", " ", text).strip()

    # 14.（可选）繁转简
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
        "含包装括号【】": sum(1 for r in rows if _STRIP_BRACKETS_RE.search(r["text"])),
        "含圈号①②③": sum(1 for r in rows if re.search(r"[" + _CIRCLED + r"]", r["text"])),
        "含异体符号(×/「/U+2011等)": sum(1 for r in rows if re.search(r"[\u2011\u2212\ufe63\u2013\u2014\uff0b\uff0a\uff5e\uff0e\uff5c\u00d7\u300c\u300d\u3014\u3015\uff3b\uff3d\uffe5\uff03\u2236\u2022\u25e6]", r["text"])),
        "含撇号类(′″〞＇')": sum(1 for r in rows if _STRIP_PRIME_RE.search(r["text"])),
        "含方括号[ ]": sum(1 for r in rows if re.search(r"[\[\]]", r["text"])),
        "含罗马数字ⅠⅡ": sum(1 for r in rows if re.search(r"[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]", r["text"])),
        "含全角字母": sum(1 for r in rows if re.search(r"[Ａ-Ｚａ-ｚ]", r["text"])),
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
        # 只输出两列：text（特征）+ category（标签）
        # 用 utf-8（不带 BOM），避免表头被读成 \ufefftext
        with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["text", "category"])
            for r in rows:
                w.writerow([r["text"], r["category"]])
        print(f"已写出（两列 text,category）: {OUT_PATH}")
    else:
        print("\n（未落盘；加 --write 生成 clean.csv）")


if __name__ == "__main__":
    main()
