# -*- coding: utf-8 -*-
"""Flask 部署（主页面，改造自 D:\\B++\\投满分.ipynb 的 Flask 单元）。

相对笔记本原版的改动：
  1) 单模型固定 -> 三个帕累托最优模型可选（ComplementNB / SGD(hinge) / BERT-NF4 4bit）；
  2) 只返回一个类别 -> 返回 Top5 类别 + 概率；
  3) 输入框改多行文本域，加清空按钮、字数计数、响应时间显示（讲义要求的两项功能）；
  4) 不可用的模型在页面上置灰并给出原因，不静默替换。

启动：
  cd deploy && python app_flask.py          # 默认 http://127.0.0.1:8000
  python app_flask.py 8800                  # 指定端口
"""
import sys

from flask import Flask, jsonify, request

from model_registry import REGISTRY
from page_html import render

app = Flask(__name__)


@app.route("/")
def index():
    return render(REGISTRY.page_models())


@app.route("/models")
def models_api():
    return jsonify(REGISTRY.page_models())


@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        data = request.get_json(force=True, silent=True) or {}
        text = (data.get("text") or "").strip()
        key = data.get("model") or "complementnb"
        if not text:
            return jsonify({"error": "文本不能为空"})
        if len(text) > 500:
            return jsonify({"error": "文本请控制在 500 字以内"})
        if key not in [m["key"] for m in REGISTRY.page_models()]:
            return jsonify({"error": f"未知模型：{key}"})
        ok, reason = REGISTRY.status[key]
        if not ok:
            return jsonify({"error": f"{REGISTRY.spec(key)['label']} 当前不可用：{reason}"})
        return jsonify(REGISTRY.predict_top5(key, text))
    except Exception as e:                      # 部署期把异常原样回给页面便于排查
        return jsonify({"error": f"预测出错：{e}"})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print("模型可用性：")
    for m in REGISTRY.page_models():
        print("   %-28s %s %s" % (m["label"], "可用" if m["available"] else "不可用",
                                  m["reason"] or ""))
    print(f"页面： http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
