# -*- coding: utf-8 -*-
"""OpenRouter 免费模型评测：把多个大模型 API 当作分类方法。

统一 prompt：给 18 个类名，要求模型只回答一个类名；解析时对返回文本做
宽松匹配（先精确命中，再子串命中，否则算错）。带重试退避以应对免费池 429。

协议与其它模型一致：按源 id 分组无泄漏划分，测试只用原始样本。
用法：
  python compare_openrouter.py                # 全量 1378 条，跑全部候选模型
  python compare_openrouter.py 100            # 分层抽样 100 条（试跑）
  python compare_openrouter.py 100 qwen/qwen3.8-27b:free   # 指定单模型
环境：从 D:\\B++\\.env 读 OPENROUTER_API_KEY
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
        "max_tokens": 512, "temperature": 0}
    last = None
    for a in range(retries):
        try:
            r = session.post(URL, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                             headers={"Authorization": "Bearer " + key,
                                      "Content-Type": "application/json",
                                      "HTTP-Referer": "https://github.com/paixiaoxin66/text_classfication",
                                      "X-Title": "text_classification eval"},
                             timeout=120)
            if r.status_code == 200:
                j = r.json()
                msg = j["choices"][0]["message"]
                content = msg.get("content") or msg.get("reasoning") or ""
                return content, None
            if r.status_code in (429, 500, 502, 503, 504):
                # 每日免费额度耗尽：重试无意义，直接抛出终止该模型
                if "per-day" in r.text or "free-models-per-day" in r.text:
                    raise RuntimeError("FREE_DAILY_QUOTA_EXCEEDED")
                # 瞬时限流：指数退避 + 抖动
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
    models = MODELS
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
    print(f"测试集 {len(test_idx)} 条（全原始）；模型数 {len(models)}", flush=True)

    rows = []
    for md in models:
        r = eval_model(key, md, test_texts, yte, cats, md.split('/')[-1])
        if r:
            rows.append(r)
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
