# -*- coding: utf-8 -*-
"""
消费者投诉文本分类数据集 —— 强标识符省流（第二阶段预处理）

背景：
    clean.csv 已完成字符级清洗。但数据中仍残留大量"强标识符"——手机号、
    统一社会信用代码、订单号/运单号/流水号等。这类标识符每条各不相同、
    零判别力、纯噪声，且上游脱敏并未做彻底（保留了 5~6 位前缀），例如：
        订单号：42000*091*          （原 4200009100000 之类）
        手机号：17888*46508
        统一社会信用代码：914*MA5FJKYQ7L

    本脚本把「整段标识符」（含已脱敏的残留前缀/后缀）省流为统一占位符，
    保留其"类型"信息（是手机号还是订单号），但丢弃无意义的数字内容。

不处理（刻意保留）：
    - 金额（如 16999元）—— 数量级跨类别有判别力
    - 日期（如 2026年9月2日）—— 有无日期的结构有判别力
    - 数量/次数/天数（如 12天、2位男士）—— 是语义量词
    - 产品型号（如 iPhone17、mate70pro）—— 是内容

用法:
    python eda/normalize_ids.py            # 只统计，不落盘
    python eda/normalize_ids.py --write    # 生成 clean_ids.csv（项目根目录）
"""
import os
import re
import csv
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV_PATH = os.path.join(ROOT, "clean.csv")
OUT_PATH = os.path.join(ROOT, "clean_ids.csv")

# ---- 占位符 ----
PH_ID = "<ID>"          # 通用标识符
PH_PHONE = "<手机号>"
PH_ORDER = "<订单号>"
PH_CODE = "<信用代码>"
PH_PLATE = "<车架号>"

# ---- 引导词 → 占位符 映射（按优先级从具体到宽泛）----
# 用于"有引导词时"决定标识符类型
LEADERS = [
    (PH_CODE,  r"统一社会信用代码|社会信用代码|信用代码|纳税人识别号|营业执照号"),
    (PH_PHONE, r"手机号码|手机号|联系电话|电话号码|联系方式|电话"),
    (PH_PLATE, r"车架号码|车架号|VIN码|VIN"),
    (PH_ORDER, r"订单号|订单编号|订单号码|订单号是|订单号为|单号|运单号|运单编号|"
               r"快递单号|物流单号|流水号|工单号|序列号|交易单号|处罚单号|"
               r"编号|条码|交易号|订单|单编|货单号|退货单号"),
]

# 标识符本体（必须以数字开头，或含星号——避免把纯字母产品词如 WiFi 当标识符）：
#   ① 含星号的脱敏串（首字符数字/字母，串中含 *），如 42000*091*  914*MA5FJKYQ7L
#   ② 无星号但较长的纯数字串（>=6 位），如 13800138000
#   ③ 数字开头、含字母的长串（>=6 字符），如 5F26QL202AI62
ID_TOKEN = (
    r"[0-9A-Za-z]*[0-9][0-9A-Za-z]*\*[0-9A-Za-z*]*"   # 含星号且含数字的脱敏串
    r"|\d{6,}"                                          # 纯数字 >=6 位
    r"|\d[0-9A-Za-z]{5,}"                               # 数字开头的长字母数字串
)

# 带引导词的完整匹配：引导词 + 可选分隔符（：: 为 是 # 空格）+ 标识符
LEADER_RE = re.compile(
    r"(?P<leader>" + "|".join(p for _, p in LEADERS) + r")"
    r"(?P<sep>\s*[:：#]?\s*|\s*为\s*|\s*是\s*)"
    r"(?P<id>" + ID_TOKEN + r")"
)

# 无引导词、但含星号的裸脱敏串（形如 数字*数字 或 *数字，长度>=5）—— 直接替换
BARE_MASKED_RE = re.compile(r"(?<![0-9A-Za-z])([0-9A-Za-z]*\*[0-9A-Za-z*]{3,}[0-9A-Za-z]|[0-9A-Za-z]{3,}\*[0-9A-Za-z*]*)(?![0-9A-Za-z])")

# 原始数据集自带的占位符 ORDER_ID（及可能尾随的 *）→ <订单号>
ORDER_ID_RE = re.compile(r"ORDER_ID\*?")


def leader_ph(leader: str) -> str:
    for ph, pat in LEADERS:
        if re.search(pat, leader):
            return ph
    return PH_ID


def normalize(text: str) -> str:
    # 1. 有引导词的标识符 → 对应占位符
    def _sub_leader(m):
        return m.group("leader") + m.group("sep") + leader_ph(m.group("leader"))
    text = LEADER_RE.sub(_sub_leader, text)

    # 1b. 原始占位符 ORDER_ID → <订单号>
    text = ORDER_ID_RE.sub(PH_ORDER, text)

    # 2. 无引导词、带星号的裸脱敏串 → 通用占位符
    def _sub_bare(m):
        return PH_ID
    text = BARE_MASKED_RE.sub(_sub_bare, text)

    return text


def load_rows(path):
    with open(path, encoding="utf-8-sig", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), reader.fieldnames


def scan(rows):
    stat = {
        "含引导词+标识符": sum(1 for r in rows if LEADER_RE.search(r["text"])),
        "含裸脱敏串(*)": sum(1 for r in rows if BARE_MASKED_RE.search(r["text"])),
        "含任意星号": sum(1 for r in rows if "*" in r["text"]),
        "含7位以上纯数字": sum(1 for r in rows if re.search(r"(?<!\d)\d{7,}(?!\d)", r["text"])),
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

    changed = 0
    samples = []
    for r in rows:
        old = r["text"]
        new = normalize(old)
        if new != old:
            changed += 1
            if len(samples) < 8:
                samples.append((old, new))
        r["text"] = new

    print(f"\n【处理结果】")
    print(f"  发生变化的样本: {changed} ({changed / len(rows) * 100:.2f}%)")

    print("\n【处理后统计】")
    for k, v in scan(rows).items():
        print(f"  {k}: {v}")

    print("\n【对照样例】")
    for old, new in samples:
        # 只截取有变化的关键片段
        print(f"  原: {old[:110]}")
        print(f"  新: {new[:110]}\n")

    if write:
        with open(OUT_PATH, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["text", "category"])
            for r in rows:
                w.writerow([r["text"], r["category"]])
        print(f"已写出（两列 text,category）: {OUT_PATH}")
    else:
        print("\n（未落盘；加 --write 生成 clean_ids.csv）")


if __name__ == "__main__":
    main()
