# -*- coding: utf-8 -*-
"""大模型 API「批量分类」评测：把多条投诉文本打包进一次请求（batch），
让模型返回 JSON（序号 -> 类别名），显著减少请求数并提升吞吐。

覆盖两家国内平台（均为 OpenAI 兼容）：
  - 商汤 SenseNova（token.sensenova.cn/v1）：sensenova-6.8-flash-lite / glm-5.2 / deepseek-v4-flash
  - 云知声 Unisound（maas.unisound.com/api/v1）：u2-flash
key 从仓库外的 D:\\B++\\.env 读取（SENSENOVA_API_KEY / U2-FLASH）。

协议与其它方法一致：按源 id 分组无泄漏划分，测试只用原始样本（1378 条）。
用法：
  python compare_llm_batch.py                        # 全量 1378 条，跑全部已确认可用模型
  python compare_llm_batch.py 128                    # 指定 batch 大小
  python compare_llm_batch.py 64 u2-flash            # 指定 batch 大小 + 单模型
  python compare_llm_batch.py 64 u2-flash glm-5.2    # 多模型
"""
import os, sys, json, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "train_augmented.csv")
SEED = 42
RESULT = os.path.join(ROOT, "resources", "tfidf_clf_result.json")

# 平台配置：key 环境变量名 + base_url
PROVIDERS = {
    "SENSENOVA_API_KEY": "https://token.sensenova.cn/v1/chat/completions",
    "U2-FLASH": "https://maas.unisound.com/api/v1/chat/completions",
}

# 已在账号下探活确认可用的模型（provider, model_id, extra_body, max_tokens）
# 关键：这些平台的「对话模型」默认开启思考，会把 token 全耗在 reasoning 上导致
# content 为空；必须用 reasoning_effort=none / thinking=disabled 关闭思考。
AVAILABLE = [
    ("SENSENOVA_API_KEY", "sensenova-6.8-flash-lite", {"reasoning_effort": "none"}, 4096),
    ("SENSENOVA_API_KEY", "glm-5.2", {"reasoning_effort": "none"}, 4096),
    ("SENSENOVA_API_KEY", "deepseek-v4-flash", {"reasoning_effort": "none"}, 4096),
    ("U2-FLASH", "u2-flash", {"thinking": {"type": "disabled"}}, 4096),
]

CATS = ["共享出行", "医疗健康", "婚恋交友", "家居日用", "影音娱乐", "房产家装", "教育",
        "数码3C", "旅游出行", "服饰鞋包", "本地生活", "母婴食品", "汽车", "游戏",
        "物流快递", "电商平台", "通讯运营商", "金融支付"]

SYS = "你是中文消费投诉文本分类助手，需要为一批文本逐条选出唯一最合适的类别。"
CATS_BLOCK = "\n".join(f"{i+1}. {c}" for i, c in enumerate(CATS))


def load_keys():
    keys = {}
    for line in open(r"D:\B++\.env", encoding="utf-8", errors="replace"):
        for k in PROVIDERS:
            if line.startswith(k):
                keys[k] = line.split("=", 1)[1].strip()
    return keys


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
    return train, te_ori, test_src


def build_prompt(texts):
    items = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
    user = (
        f"候选类别（每条只能选一个，必须严格照抄类别名）：\n{CATS_BLOCK}\n\n"
        f"请为下面每条投诉文本分类，只输出一个 JSON 对象，键为文本序号（从 1 开始），"
        f"值为类别名，不要输出任何解释或多余文字：\n{items}\n\n"
        f'只输出 JSON，例如：{{"1": "电商平台", "2": "物流快递"}}'
    )
    return user


def extract_json(content):
    """从模型输出里抠出 {序号: 类别名} 字典。"""
    if not content:
        return {}
    s = content.strip()
    # 去 markdown 代码围栏
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.S)
    if m:
        s = m.group(1)
    # 直接找第一个平衡的 {...}
    start = s.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    s = s[start:i+1]
                    break
    try:
        obj = json.loads(s)
    except Exception:
        return {}
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                out[str(k).strip()] = v.strip()
    return out


def match_cat(v):
    """把模型给的类别名对齐到 18 类之一。"""
    if not v:
        return None
    c = v.strip().strip("。.，,、！!？?\"'“”‘’「」【】()（） \n")
    if c in CATS:
        return c
    # 简繁/异体归一：按单字替换后再试
    # 简繁/异体归一：按单字替换后再试（覆盖 18 类名可能出现的繁体写法）
    TRAD = {
        "產": "产", "貨": "货", "電": "电", "腦": "脑", "網": "网", "絡": "络", "園": "园",
        "專": "专", "發": "发", "車": "车", "設": "设", "備": "备", "機": "机", "體": "体",
        "育": "育", "樂": "乐", "裝": "装", "飾": "饰", "訊": "讯", "運": "运", "營": "营",
        "醫": "医", "療": "疗", "健": "健", "康": "康", "龍": "龙", "鮮": "鲜", "嬰": "婴",
        "兒": "儿", "郵": "邮", "遞": "递", "險": "险", "費": "费", "戶": "户", "數": "数",
        "碼": "码", "遊": "游", "戲": "戏", "娛": "娱", "學": "学", "習": "习", "業": "业",
        "產": "产", "觀": "观", "視": "视", "頻": "频", "飲": "饮", "飾": "饰", "類": "类",
        "諮": "咨", "詢": "询", "溝": "沟", "統": "统", "際": "际", "絡": "络", "鏈": "链",
        "點": "点", "評": "评", "訴": "诉", "務": "务", "質": "质", "檢": "检", "驗": "验",
    }
    tr = "".join(TRAD.get(ch, ch) for ch in c)
    if tr in CATS:
        return tr
    hits = [k for k in CATS if k in v or k in tr]
    if hits:
        return max(hits, key=len)
    low = (v + tr).lower()
    for k in CATS:
        if k.lower() in low:
            return k
    return None


def ask_batch(session, url, key, model, user, max_tokens, extra_body=None, retries=5):
    body = {"model": model, "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 0}
    if extra_body:
        body.update(extra_body)
    last = None
    for a in range(retries):
        try:
            r = session.post(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                             headers={"Authorization": "Bearer " + key,
                                      "Content-Type": "application/json"},
                             timeout=300)
            if r.status_code == 200:
                j = r.json()
                msg = j["choices"][0]["message"]
                content = msg.get("content") or msg.get("reasoning_content") or msg.get("reasoning") or ""
                if not content.strip():
                    last = "empty_content"
                    continue  # 空内容（思考耗尽 token）→ 重试
                return content, None
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(min(30, 3.0 * (a + 1)) + np.random.rand())
                last = f"HTTP{r.status_code}:{r.text[:100]}"
                continue
            return None, f"HTTP{r.status_code}:{r.text[:140]}"
        except Exception as e:
            time.sleep(min(30, 3.0 * (a + 1)))
            last = repr(e)[:140]
    return None, last


def eval_block(S, url, key, model, texts, yte, idxs, max_tokens, extra_body, preds, depth=0):
    """评估一个批次；若整批解析失败，则二分再试（递归），提升鲁棒性。"""
    n_bad = 0
    content, err = ask_batch(S, url, key, model,
                             build_prompt([texts[j] for j in idxs]), max_tokens, extra_body)
    obj = extract_json(content)
    got = 0
    if obj:
        for pos, j in enumerate(idxs, start=1):
            ps = match_cat(obj.get(str(pos)))
            if ps is not None:
                preds[j] = ps
                got += 1
    # 未解析/漏答太多 → 二分重试（最多 3 层）
    missing = [j for j in idxs if preds[j] is None]
    if missing and depth < 3 and (not obj or len(missing) > len(idxs) * 0.2):
        mid = len(idxs) // 2
        if mid == 0:
            return len(idxs)
        a = eval_block(S, url, key, model, texts, yte, idxs[:mid], max_tokens, extra_body, preds, depth + 1)
        b = eval_block(S, url, key, model, texts, yte, idxs[mid:], max_tokens, extra_body, preds, depth + 1)
        return a + b
    return len(missing)


def eval_model(keys, provider, model, extra_body, max_tokens, texts, yte, batch_size, tag):
    url = PROVIDERS[provider]
    key = keys[provider]
    n = len(texts)
    blocks = [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]
    t0 = time.time()
    preds = [None] * n
    # SenseNova 免费套餐 RPM 较低，降到 2 并发避免级联 429；云知声可稍高
    workers = 2 if provider == "SENSENOVA_API_KEY" else 4
    with requests.Session() as S:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(eval_block, S, url, key, model, texts, yte, blk,
                              max_tokens, extra_body, preds): blk for blk in blocks}
            done = 0
            for f in as_completed(futs):
                f.result()
                done += 1
                if done % 5 == 0:
                    ok = sum(p is not None for p in preds)
                    print(f"    [{tag}] 批次 {done}/{len(blocks)} 用时{time.time()-t0:.0f}s "
                          f"已解析{ok}/{n}", flush=True)
    dt = time.time() - t0
    ok = sum(p is not None for p in preds)
    # 未解析的按"未命中"处理：对解析成功的子集评分（与其它 API 方法一致口径）
    mask = [p is not None for p in preds]
    ys = [yte[j] for j in range(n) if mask[j]]
    ps = [preds[j] for j in range(n) if mask[j]]
    if not ys:
        print(f"  {tag:34s} 全部解析失败", flush=True)
        return None
    f1 = f1_score(ys, ps, average="macro", labels=CATS, zero_division=0)
    acc = accuracy_score(ys, ps)
    print(f"  {tag:34s} macroF1={f1:.4f} acc={acc:.4f} 用时{dt:.0f}s "
          f"有效{ok}/{n} 解析失败{n-ok}", flush=True)
    return dict(name=f"LLM:{model}", macro_f1=round(float(f1), 4), acc=round(float(acc), 4),
                fit_s=None, infer_s=round(dt, 1),
                extra=f"batch{batch_size},n={ok},{n-ok}fail,api")


def main():
    args = sys.argv[1:]
    batch_size = 64
    models = AVAILABLE
    if args and args[0].isdigit():
        batch_size = int(args[0]); args = args[1:]
    if args:
        # 用户指定模型名（按名匹配 AVAILABLE）
        sel = []
        for a in args:
            hit = [m for m in AVAILABLE if m[1] == a]
            if hit:
                sel += hit
            else:
                print(f"警告：模型 {a} 不在已确认可用列表，跳过")
        if sel:
            models = sel

    keys = load_keys()
    df = pd.read_csv(CSV)
    cats = sorted(df["category"].unique())
    y = df["category"].map({c: i for i, c in enumerate(cats)}).to_numpy()
    texts = df["text"].astype(str).tolist()
    train, test, test_src = build_split(df)
    test_idx = list(test)
    yte = [cats[y[i]] for i in test_idx]
    test_texts = [texts[i] for i in test_idx]
    print(f"测试集 {len(test_idx)} 条（全原始）；batch={batch_size}；模型数 {len(models)}", flush=True)

    rows = []
    for provider, md, extra_body, max_tokens in models:
        r = eval_model(keys, provider, md, extra_body, max_tokens,
                       test_texts, yte, batch_size, md)
        if r:
            rows.append(r)
    rows.sort(key=lambda r: -r["macro_f1"])
    print("\n=== 本批排名 ===")
    for r in rows:
        print(f"  {r['name']:36s} F1={r['macro_f1']:.4f} acc={r['acc']:.4f} {r['infer_s']}s")

    if rows:
        prev = json.load(open(RESULT, encoding="utf-8")) if os.path.exists(RESULT) else {"rows": []}
        prev_rows = prev.get("rows", [])
        names = {r["name"] for r in rows}
        merged = [r for r in prev_rows if r["name"] not in names] + rows
        merged.sort(key=lambda r: -r["macro_f1"])
        prev["rows"] = merged
        prev["best"] = merged[0] if merged else None
        json.dump(prev, open(RESULT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("已写:", RESULT, flush=True)


if __name__ == "__main__":
    main()
