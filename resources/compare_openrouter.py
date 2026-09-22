# -*- coding: utf-8 -*-
"""OpenRouter 免费模型评测：把多个大模型 API 当作分类方法。

统一 prompt：给 18 个类名，要求模型只回答一个类名；解析时对返回文本做
宽松匹配（先精确命中，再子串命中，否则算错）。带重试退避以应对免费池 429。

协议与其它模型一致：按源 id 分组无泄漏划分，测试只用原始样本。
用法：
  python compare_openrouter.py                # 全量 1378 条，跑全部候选模型（单条请求）
  python compare_openrouter.py 100            # 分层抽样 100 条（试跑）
  python compare_openrouter.py 100 qwen/qwen3.8-27b:free   # 指定单模型
  python compare_openrouter.py --batch=64     # 全量 1378 条，但每 64 条打包成一次请求
环境：从 D:\\B++\\.env 读 OPENROUTER_API_KEY

关于 batch 模式（重要）：
  免费池每日上限仅 50 次请求（free_model_daily_requests，0 credits 时固定 50）。
  单条 1 请求全量需 1378 次，永远无法在一天内跑完；batch=64 时每个模型只要
  22 次请求，与其它 LLM 行（batch64,n=1378,api）口径一致。
"""
import os, sys, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, accuracy_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(ROOT, "train_augmented.csv")
SEED = 42
URL = "https://openrouter.ai/api/v1/chat/completions"

# 候选免费模型（挑通用/中文能力强者，排除 code/安全/VL 专用）
# 注：thinkingmachines/inkling:free 仅限 agentic harness，普通 chat 调用返回 403，故不列入。
MODELS = [
    "nex-agi/nex-n2.5-pro:free",
    "qwen/qwen3.8-27b:free",
    "z-ai/glm-5.2:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nex-agi/nex-n2.5-mini:free",
    "dots-studio/dots-3-note-preview:free",
    "inclusionai/ling-3.0-flash-fin:free",
]

CATS = ["共享出行", "医疗健康", "婚恋交友", "家居日用", "影音娱乐", "房产家装", "教育",
        "数码3C", "旅游出行", "服饰鞋包", "本地生活", "母婴食品", "汽车", "游戏",
        "物流快递", "电商平台", "通讯运营商", "金融支付"]

SYS = ("你是中文消费投诉文本分类助手。用户给出一条投诉文本和一个候选类别列表，"
       "你必须从中选出唯一最合适的一个类别，只输出该类别的名称本身，"
       "不要输出任何标点、解释或多余文字。")
USER_TPL = ("候选类别（只能选一个）：\n{cats}\n\n"
            "投诉文本：\n{text}\n\n"
            "请只输出最合适的一个类别名。")


def load_key():
    for line in open(r"D:\B++\.env", encoding="utf-8", errors="replace"):
        if line.startswith("OPENROUTER_API_KEY"):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("OPENROUTER_API_KEY not found")


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


def parse_choice(content, cats):
    """宽松解析：精确 -> 去空白后的精确 -> 子串包含（取最长命中）。"""
    if not content:
        return None
    c = content.strip().strip("。.，,、！!？?\"'“”‘’「」【】()（） \n")
    if c in cats:
        return c
    hits = [k for k in cats if k in content]
    if hits:
        return max(hits, key=len)
    # 处理 3C 之类大小写/全半角
    low = content.lower()
    for k in cats:
        if k.lower() in low:
            return k
    return None


def ask(session, key, model, text, cats_str, retries=6):
    body = {"model": model, "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": USER_TPL.format(cats=cats_str, text=text)}],
        "max_tokens": 512, "temperature": 0,
        # 免费池里不少是思考模型（nex/nemotron 等）：不关思考会把 token 全烧在
        # reasoning 上，content 返回空 → 判为解析失败。enabled=false 后 content 直出。
        "reasoning": {"enabled": False}}
    return _post(session, key, body, retries)


def _post(session, key, body, retries=6):
    last = None
    for a in range(retries):
        try:
            CALLS[0] += 1
            r = session.post(URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                             headers={"Authorization": "Bearer " + key,
                                      "Content-Type": "application/json",
                                      "HTTP-Referer": "https://github.com/Barry-Wuu/text_classfication",
                                      "X-Title": "text_classification eval"},
                             timeout=300)
            if r.status_code == 200:
                j = r.json()
                msg = j["choices"][0]["message"]
                content = msg.get("content") or msg.get("reasoning") or ""
                if not content.strip():
                    last = "empty_content"
                    time.sleep(min(10, 2.0 * (a + 1)))
                    continue
                return content, None
            if r.status_code in (429, 500, 502, 503, 504):
                # 每日免费额度耗尽：重试无意义，直接抛出终止该模型
                if "per-day" in r.text or "free-models-per-day" in r.text:
                    raise RuntimeError("FREE_DAILY_QUOTA_EXCEEDED")
                # 瞬时限流（含上游共享池 upstream_provider_shared_pool）：指数退避 + 抖动
                time.sleep(min(20, 2.0 * (a + 1)) + np.random.rand() * 0.5)
                last = f"HTTP{r.status_code}"
                continue
            return None, f"HTTP{r.status_code}:{r.text[:120]}"
        except RuntimeError:
            raise
        except Exception as e:
            time.sleep(min(20, 2.0 * (a + 1)))
            last = repr(e)[:120]
    return None, last


# ---------------- batch 模式：多条文本打包进一次请求 ----------------
CALLS = [0]  # 实际 API 请求计数（含重试），用于汇报额度消耗

BATCH_SYS = "你是中文消费投诉文本分类助手，需要为一批文本逐条选出唯一最合适的类别。"
BATCH_USER_TPL = ("候选类别（每条只能选一个，必须严格照抄类别名）：\n{cats}\n\n"
                  "请为下面每条投诉文本分类，只输出一个 JSON 对象，键为文本序号（从 1 开始），"
                  "值为类别名，不要输出任何解释或多余文字：\n{items}\n\n"
                  '只输出 JSON，例如：{{"1": "电商平台", "2": "物流快递"}}')


def extract_json(content):
    if not content:
        return {}
    import re
    s = content.strip()
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.S)
    if m:
        s = m.group(1)
    start = s.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    s = s[start:i + 1]
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


def build_batch_prompt(texts, cats):
    cats_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(cats))
    items = "\n".join(f"[{i+1}] {t}" for i, t in enumerate(texts))
    return BATCH_USER_TPL.format(cats=cats_block, items=items)


def eval_block(session, key, model, texts, cats, idxs, preds, depth=0):
    """评估一个批次；整批解析失败或漏答过多则二分重试（最多 3 层）。"""
    content, err = _post(session, key,
                         {"model": model, "messages": [
                             {"role": "system", "content": BATCH_SYS},
                             {"role": "user", "content": build_batch_prompt([texts[j] for j in idxs], cats)}],
                          "max_tokens": 4096, "temperature": 0,
                          "reasoning": {"enabled": False}})
    obj = extract_json(content)
    if obj:
        for pos, j in enumerate(idxs, start=1):
            p = parse_choice(obj.get(str(pos)), cats)
            if p is not None:
                preds[j] = p
    missing = [j for j in idxs if preds[j] is None]
    if missing and depth < 3 and (not obj or len(missing) > len(idxs) * 0.2):
        mid = len(idxs) // 2
        if mid == 0:
            return len(missing)
        a = eval_block(session, key, model, texts, cats, idxs[:mid], preds, depth + 1)
        b = eval_block(session, key, model, texts, cats, idxs[mid:], preds, depth + 1)
        return a + b
    return len(missing)


def eval_model_batch(key, model, texts, yte, cats, tag, batch_size):
    n = len(texts)
    blocks = [list(range(i, min(i + batch_size, n))) for i in range(0, n, batch_size)]
    t0 = time.time()
    preds = [None] * n
    abort = False
    with requests.Session() as S:
        with ThreadPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(eval_block, S, key, model, texts, cats, blk, preds): blk
                    for blk in blocks}
            done = 0
            for f in as_completed(futs):
                try:
                    f.result()
                except RuntimeError as e:
                    if "FREE_DAILY_QUOTA_EXCEEDED" in str(e):
                        abort = True
                        for ff in futs:
                            ff.cancel()
                        break
                done += 1
                if done % 5 == 0:
                    ok = sum(p is not None for p in preds)
                    print(f"    [{tag}] 批次{done}/{len(blocks)} 用时{time.time()-t0:.0f}s "
                          f"已解析{ok}/{n}", flush=True)
    if abort:
        print(f"  {tag:42s} 跳过：免费模型每日额度已耗尽（free-models-per-day）", flush=True)
        return None
    dt = time.time() - t0
    mask = [p is not None for p in preds]
    ys = [yte[j] for j in range(n) if mask[j]]
    ps = [preds[j] for j in range(n) if mask[j]]
    if not ys:
        return None
    f1 = f1_score(ys, ps, average="macro", labels=cats, zero_division=0)
    acc = accuracy_score(ys, ps)
    print(f"  {tag:42s} macroF1={f1:.4f} acc={acc:.4f} 用时{dt:.0f}s "
          f"有效{sum(mask)}/{n} 解析失败{n-sum(mask)}", flush=True)
    return dict(name=f"LLM:{model.split('/')[-1]}", macro_f1=round(float(f1), 4),
                acc=round(float(acc), 4), fit_s=None, infer_s=round(dt, 1),
                extra=f"batch{batch_size},n={sum(mask)},{n-sum(mask)}fail,api")


def eval_model(key, model, texts, yte, cats, tag):
    cats_str = "、".join(cats)
    t0 = time.time()
    preds = [None] * len(texts)
    fails = 0
    abort = False
    with requests.Session() as S:
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(ask, S, key, model, texts[j], cats_str): j for j in range(len(texts))}
            done = 0
            for f in as_completed(futs):
                j = futs[f]
                try:
                    content, err = f.result()
                except RuntimeError as e:
                    if "FREE_DAILY_QUOTA_EXCEEDED" in str(e):
                        abort = True
                        for ff in futs:
                            ff.cancel()
                        break
                    content, err = None, str(e)
                if content is None:
                    fails += 1; preds[j] = None
                else:
                    preds[j] = parse_choice(content, cats)
                done += 1
                if done % 50 == 0:
                    print(f"    [{tag}] {done}/{len(texts)} 用时{time.time()-t0:.0f}s 失败{fails}", flush=True)
    if abort:
        print(f"  {tag:42s} 跳过：免费模型每日额度已耗尽（free-models-per-day）", flush=True)
        return None
    dt = time.time() - t0
    mask = [p is not None for p in preds]
    ys = [yte[j] for j in range(len(texts)) if mask[j]]
    ps = [preds[j] for j in range(len(texts)) if mask[j]]
    if not ys:
        return None
    f1 = f1_score(ys, ps, average="macro", labels=cats, zero_division=0)
    acc = accuracy_score(ys, ps)
    print(f"  {tag:42s} macroF1={f1:.4f} acc={acc:.4f} 用时{dt:.0f}s "
          f"有效{sum(mask)}/{len(texts)} 解析失败{len(texts)-sum(mask)}", flush=True)
    return dict(name=f"LLM:{model.split('/')[-1]}", macro_f1=round(float(f1), 4),
                acc=round(float(acc), 4), fit_s=None, infer_s=round(dt, 1),
                extra=f"n={sum(mask)},{len(texts)-sum(mask)}fail,api")


def main():
    key = load_key()
    df = pd.read_csv(CSV)
    cats = sorted(df["category"].unique())
    y = df["category"].map({c: i for i, c in enumerate(cats)}).to_numpy()
    texts = df["text"].astype(str).tolist()
    train, test, test_src = build_split(df)

    args = sys.argv[1:]
    limit = 0
    batch = 0
    models = MODELS
    rest = []
    for a in args:
        if a.startswith("--batch="):
            batch = int(a.split("=", 1)[1])
        else:
            rest.append(a)
    args = rest
    if args:
        if args[0].isdigit():
            limit = int(args[0]); args = args[1:]
        if args:
            models = args  # 指定模型
    test_idx = list(test)
    if limit and limit < len(test_idx):
        rng = np.random.default_rng(SEED)
        per = {}
        for i in test_idx:
            per.setdefault(y[i], []).append(i)
        picked = []
        for k, arr in per.items():
            rng.shuffle(arr)
            picked += arr[:max(1, round(limit * len(arr) / len(test_idx)))]
        test_idx = sorted(picked)[:limit]
    yte = [cats[y[i]] for i in test_idx]
    test_texts = [texts[i] for i in test_idx]
    print(f"测试集 {len(test_idx)} 条（全原始）；模型数 {len(models)}；"
          f"batch={batch if batch else '单条'}", flush=True)

    rows = []
    for md in models:
        if batch:
            r = eval_model_batch(key, md, test_texts, yte, cats, md.split('/')[-1], batch)
        else:
            r = eval_model(key, md, test_texts, yte, cats, md.split('/')[-1])
        if r:
            rows.append(r)
    print(f"实际 API 请求数（含重试）：{CALLS[0]}", flush=True)
    rows.sort(key=lambda r: -r["macro_f1"])
    print("\n=== 本批排名 ===")
    for r in rows:
        print(f"  {r['name']:40s} F1={r['macro_f1']:.4f} acc={r['acc']:.4f}")

    if not limit or limit >= len(test):
        p = os.path.join(ROOT, "resources", "tfidf_clf_result.json")
        prev = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        prev_rows = prev.get("rows", [])
        names = {r["name"] for r in rows}
        merged = [r for r in prev_rows if r["name"] not in names] + rows
        merged.sort(key=lambda r: -r["macro_f1"])
        prev["rows"] = merged; prev["best"] = merged[0] if merged else None
        json.dump(prev, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("已写:", p, flush=True)
    else:
        print("（抽样试跑，未写入正式结果文件）", flush=True)


if __name__ == "__main__":
    main()
