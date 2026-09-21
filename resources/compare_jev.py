# -*- coding: utf-8 -*-
"""TypeSafe Jev（System One 模型）评测：把大模型 API 当作一种分类方法。

Jev 是 TypeSafe 的 System One 模型，用 Choice 原语直接做结构化分类：
  - criteria = 18 个类名（英文 key? 不——直接用中文类名作 key）
  - 返回 choice / probabilities / confidence，无需解析文本

协议与其它模型完全一致：按源 id 分组无泄漏划分，测试只用原始样本。
默认跑全量测试集（1378 条），并发请求（线程池）。

用法：
  python compare_jev.py             # 全量 1378 条
  python compare_jev.py 100         # 只跑前 100 条（分层抽样）便于试跑
环境：从 D:\\B++\\.env 读 TYPESAFE
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
API = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

# 18 类的判定说明（帮助 Jev 区分相邻类别）
CRITERIA = {
    "共享出行": "共享单车、共享充电宝、网约车、共享汽车等共享服务的费用或服务纠纷",
    "医疗健康": "医院、诊所、药品、体检、医美、健康服务相关投诉",
    "婚恋交友": "婚恋平台、相亲、交友 APP 的会员或服务纠纷",
    "家居日用": "家具、家纺、日用百货、厨具等家居商品投诉",
    "影音娱乐": "电影、演出、视频/音乐平台会员、KTV 等娱乐消费",
    "房产家装": "买房、租房、中介、装修、物业相关纠纷",
    "教育": "培训机构、在线课程、学费退费等教育消费",
    "数码3C": "手机、电脑、相机、智能设备等数码产品投诉",
    "旅游出行": "旅行社、酒店、机票、景区门票等旅游出行消费",
    "服饰鞋包": "服装、鞋子、箱包、配饰等商品投诉",
    "本地生活": "外卖、餐饮、到店团购、本地生活服务",
    "母婴食品": "母婴用品、奶粉、婴儿食品相关投诉",
    "汽车": "汽车购买、维修、保养、4S 店相关纠纷",
    "游戏": "网络游戏充值、账号、道具、封号等纠纷",
    "物流快递": "快递、物流、配送延误、丢件、损坏等",
    "电商平台": "淘宝京东拼多多等电商平台的订单、售后、平台规则纠纷",
    "通讯运营商": "手机套餐、流量、话费、宽带、运营商服务",
    "金融支付": "银行卡、支付、贷款、理财、保险等金融产品纠纷",
}


def load_key():
    for line in open(r"D:\B++\.env", encoding="utf-8", errors="replace"):
        if line.startswith("TYPESAFE"):
            return line.split("=", 1)[1].strip()
    raise RuntimeError("TYPESAFE not found in .env")


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


def ask_jev(session, key, text, retries=3):
    body = {
        "state": str(text),
        "model": MODEL,
        "questions": {
            "category": {
                "type": "choice",
                "instructions": "这条消费者投诉属于下面哪一个消费领域",
                "criteria": CRITERIA,
            }
        },
    }
    for a in range(retries):
        try:
            r = session.post(API, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                             headers={"Authorization": "Bearer " + key,
                                      "Content-Type": "application/json"}, timeout=90)
            if r.status_code == 200:
                ans = r.json()["answers"]["category"]
                return ans.get("choice"), ans.get("confidence")
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.5 * (a + 1)); continue
            return None, f"HTTP{r.status_code}:{r.text[:120]}"
        except Exception as e:
            time.sleep(1.5 * (a + 1))
            last = repr(e)[:120]
    return None, last


def main():
    key = load_key()
    df = pd.read_csv(CSV)
    cats = sorted(df["category"].unique())
    assert set(cats) == set(CRITERIA.keys()), (len(cats), len(CRITERIA))
    m = {c: i for i, c in enumerate(cats)}
    y = df["category"].map(m).to_numpy()
    texts = df["text"].astype(str).tolist()
    train, test, test_src = build_split(df)

    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    test_idx = list(test)
    if limit and limit < len(test_idx):
        # 分层抽样：按 gold 类别等比例取
        rng = np.random.default_rng(SEED)
        per = {}
        for i in test_idx:
            per.setdefault(y[i], []).append(i)
        picked = []
        for k, arr in per.items():
            rng.shuffle(arr)
            picked += arr[:max(1, round(limit * len(arr) / len(test_idx)))]
        test_idx = sorted(picked)[:limit]
    yte = y[test_idx]
    print(f"测试集 {len(test_idx)} 条（全原始）；gold 类别数 {len(set(yte))}", flush=True)

    y_real = [cats[i] for i in yte]
    t0 = time.time()
    results = [None] * len(test_idx)
    fails = []
    with requests.Session() as S:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futs = {ex.submit(ask_jev, S, key, texts[i]): j for j, i in enumerate(test_idx)}
            done = 0
            for f in as_completed(futs):
                j = futs[f]
                choice, conf = f.result()
                results[j] = (choice, conf)
                if choice is None:
                    fails.append(conf)
                done += 1
                if done % 50 == 0:
                    print(f"  ...{done}/{len(test_idx)} 用时{time.time()-t0:.0f}s", flush=True)
    dt = time.time() - t0

    ok = [(r, g) for r, g in zip(results, y_real) if r[0] is not None]
    preds = [cats.index(r[0]) if r[0] in cats else -1 for r, g in zip(results, y_real)]
    mask = [p >= 0 for p in preds]
    ys = [y_real[k] for k, m_ in enumerate(mask) if m_]
    ps = [cats[preds[k]] for k, m_ in enumerate(mask) if m_]
    f1 = f1_score(ys, ps, average="macro")
    acc = accuracy_score(ys, ps)
    avg_conf = np.mean([r[1] for r, g in zip(results, y_real) if isinstance(r[1], float)])

    print(f"\n  Jev(大模型API)  macroF1={f1:.4f} acc={acc:.4f} "
          f"总耗时={dt:.1f}s 有效={len(ok)}/{len(test_idx)} 平均置信={avg_conf:.3f}", flush=True)
    if fails:
        print("  失败样例:", fails[:5], flush=True)

    rows = [dict(name="Jev(LLM-API)", macro_f1=round(float(f1), 4), acc=round(float(acc), 4),
                 fit_s=round(dt, 1), infer_s=round(dt, 1), extra=f"n={len(test_idx)}")]
    print("\n=== 排名 ===")
    for r in rows:
        print(f"  {r['name']:16s} F1={r['macro_f1']:.4f} acc={r['acc']:.4f}")

    # 只有全量时才并入正式结果文件
    if not limit or limit >= len(test):
        p = os.path.join(ROOT, "resources", "tfidf_clf_result.json")
        prev = json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        prev_rows = prev.get("rows", [])
        merged = [r for r in prev_rows if r["name"] != "Jev(LLM-API)"] + rows
        merged.sort(key=lambda r: -r["macro_f1"])
        prev["rows"] = merged
        prev["best"] = merged[0] if merged else None
        json.dump(prev, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print("已写:", p, flush=True)
    else:
        print("（抽样试跑，未写入正式结果文件）", flush=True)


if __name__ == "__main__":
    main()
