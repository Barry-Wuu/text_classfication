# -*- coding: utf-8 -*-
"""部署用模型注册表：统一 Top5 预测接口。

三个帕累托最优模型（5.6 节前沿成员）：
  complementnb  ComplementNB（CPU，3.5MB）
  sgd_hinge     SGD（hinge）（CPU，1.5MB）
  bert_nf4      BERT-NF4 4bit 量化（GPU，108MB，需 bitsandbytes）

附赠一个部署上更实用的 CPU 版 BERT：
  bert_int8     BERT-INT8 动态量化（CPU，146MB）

约定：
  - 概率：sklearn 有 predict_proba 直接用；SGD(hinge) 没有 predict_proba，
    用 OvR decision_function 经 softmax 近似（与 5.7 节软投票的处理一致）。
  - 不可用的模型不静默替换，而是给出原因（页面据此置灰）。
"""
import json
import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
BASE_BERT = os.environ.get("BASE_BERT", r"D:\models\bert-base-chinese")

LABELS = json.load(open(os.path.join(MODEL_DIR, "labels.json"), encoding="utf-8"))
N_LABELS = len(LABELS)

# 展示用元信息（指标取自 5.5 / 5.11 的实测值）
SPECS = [
    dict(key="complementnb", label="ComplementNB", device="CPU",
         macro_f1="0.8068", size_mb=3.5, kind="sklearn",
         note="朴素贝叶斯，0.1 秒训练，秒级推理"),
    dict(key="sgd_hinge", label="SGD（hinge）", device="CPU",
         macro_f1="0.8138", size_mb=1.5, kind="sklearn",
         note="线性 SVM 的随机梯度版，1.5 秒训练"),
    dict(key="bert_nf4", label="BERT-NF4 4bit 量化", device="GPU",
         macro_f1="0.8590", size_mb=108.1, kind="bert_nf4",
         note="体积 -72%、GPU 推理 3.1x，需 CUDA + bitsandbytes"),
    dict(key="bert_int8", label="BERT-INT8 动态量化", device="CPU",
         macro_f1="0.8518", size_mb=145.6, kind="bert_int8",
         note="纯 CPU 上的 BERT，动态量化"),
]
DEFAULT_KEY = "complementnb"


def _softmax(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


class SklearnModel:
    def __init__(self, key):
        import joblib
        from jieba_cut import cut          # noqa: F401  供 joblib 反序列化引用
        self.key = key
        self.vec = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib"))
        self.clf = joblib.load(os.path.join(MODEL_DIR, key + ".joblib"))
        self._cut = cut

    def predict_proba(self, text):
        # 向量化器自带 tokenizer（jieba_cut.cut），这里必须传原文，不能先分词
        x = self.vec.transform([text])
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(x)[0]
        # SGD(hinge) 无 predict_proba：OvR decision_function -> softmax
        return _softmax(self.clf.decision_function(x)[0])


class BertModel:
    """NF4 4bit / INT8 动态量化两种 BERT 变体共用加载逻辑。"""

    def __init__(self, key, ckpt_path, quant="nf4"):
        import torch
        from transformers import BertTokenizerFast, BertForSequenceClassification
        self.torch = torch
        self.key = key
        self.quant = quant
        self.tok = BertTokenizerFast.from_pretrained(BASE_BERT)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
        n = int(torch.load(ckpt_path, map_location="cpu",
                           weights_only=False).get("num_labels", N_LABELS))
        model = BertForSequenceClassification.from_pretrained(BASE_BERT, num_labels=n)
        if quant == "nf4":
            if not torch.cuda.is_available():
                raise RuntimeError("本机无 CUDA")
            import bitsandbytes as bnb
            for name, mod in list(model.named_modules()):
                if isinstance(mod, torch.nn.Linear) and ".encoder." in name + ".":
                    parent = model.get_submodule(name.rsplit(".", 1)[0])
                    setattr(parent, name.rsplit(".", 1)[1],
                            bnb.nn.Linear4bit(mod.in_features, mod.out_features,
                                              bias=mod.bias is not None,
                                              compute_dtype=torch.float16,
                                              quant_type="nf4"))
            model.load_state_dict(state, strict=False)
            model.to("cuda").eval()
        else:
            model = torch.ao.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8)
            model.load_state_dict(state)
            model.eval()
        self.model = model

    def predict_proba(self, text):
        torch = self.torch
        enc = self.tok([text], max_length=128, truncation=True, padding=True,
                       return_tensors="pt")
        dev = "cuda" if self.quant == "nf4" else "cpu"
        with torch.no_grad():
            out = self.model(input_ids=enc["input_ids"].to(dev),
                             attention_mask=enc["attention_mask"].to(dev))
            logits = out.logits[0].float().cpu().numpy()
        return _softmax(logits)


def ckpt_path(key):
    """量化权重位置：优先 deploy/models/，其次环境变量指定的目录。"""
    local = os.path.join(MODEL_DIR, f"{key}_state.pt")
    ext = os.environ.get("DEPLOY_MODEL_DIR", "")
    if os.path.exists(local):
        return local
    if ext and os.path.exists(os.path.join(ext, f"{key}_state.pt")):
        return os.path.join(ext, f"{key}_state.pt")
    return local


class Registry:
    def __init__(self):
        self.cache = {}
        self.status = {}
        for s in SPECS:
            self.status[s["key"]] = self._probe(s)

    def _probe(self, spec):
        """不加载模型，只判断是否可用（避免启动就吃掉几 GB 内存）。"""
        k, kind = spec["key"], spec["kind"]
        if kind == "sklearn":
            p = os.path.join(MODEL_DIR, k + ".joblib")
            return (os.path.exists(p), "" if os.path.exists(p) else "缺少模型文件")
        p = ckpt_path(k)
        if not os.path.exists(p):
            return False, "未下载权重"
        if kind == "bert_nf4":
            try:
                import torch
                if not torch.cuda.is_available():
                    return False, "本机无 CUDA"
                import bitsandbytes  # noqa: F401
            except Exception:
                return False, "缺 bitsandbytes"
        return True, ""

    def spec(self, key):
        for s in SPECS:
            if s["key"] == key:
                return s
        raise KeyError(key)

    def _load(self, key):
        if key in self.cache:
            return self.cache[key]
        spec = self.spec(key)
        ok, reason = self.status[key]
        if not ok:
            raise RuntimeError(reason)
        if spec["kind"] == "sklearn":
            obj = SklearnModel(key)
        else:
            obj = BertModel(key, ckpt_path(key),
                            quant="nf4" if spec["kind"] == "bert_nf4" else "int8")
        self.cache[key] = obj
        return obj

    def predict_top5(self, key, text, k=5):
        model = self._load(key)
        t0 = time.perf_counter()
        probs = np.asarray(model.predict_proba(text), dtype=np.float64)
        ms = (time.perf_counter() - t0) * 1000.0
        order = np.argsort(-probs)[:k]
        return {
            "model": key,
            "model_label": self.spec(key)["label"],
            "device": self.spec(key)["device"],
            "ms": round(ms, 2),
            "top5": [{"label": LABELS[i], "prob": float(probs[i])} for i in order],
        }

    def page_models(self):
        """给页面用的模型清单（含可用性与不可用原因）。"""
        out = []
        for s in SPECS:
            ok, reason = self.status[s["key"]]
            out.append(dict(key=s["key"], label=s["label"], device=s["device"],
                            macro_f1=s["macro_f1"], size_mb=s["size_mb"],
                            available=bool(ok), reason=reason,
                            default=(s["key"] == DEFAULT_KEY)))
        return out


REGISTRY = Registry()

if __name__ == "__main__":
    print(json.dumps(REGISTRY.page_models(), ensure_ascii=False, indent=1))
    demo = "在拼多多买的运动鞋开胶，商家拒绝退货，申请平台介入无果"
    for k in ("complementnb", "sgd_hinge"):
        r = REGISTRY.predict_top5(k, demo)
        print(f"\n[{k}] {r['ms']} ms")
        for it in r["top5"]:
            print("   %-8s %.4f" % (it["label"], it["prob"]))
