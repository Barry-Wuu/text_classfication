# -*- coding: utf-8 -*-
"""FastAPI 部署（讲义三种部署方式里最适合做服务化的版本）。

- 与 Flask 版共用同一张页面（deploy/page_html.py）与同一份模型注册表；
- 额外提供 /models 与 /predict 的 OpenAPI 文档（/docs 自动生成）。

启动：
  cd deploy && python app_fastapi.py        # http://127.0.0.1:8001
  python app_fastapi.py 8801
"""
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from model_registry import REGISTRY
from page_html import render

app = FastAPI(title="投满分 · 消费者投诉文本分类", version="1.0")


class PredictIn(BaseModel):
    text: str = Field(..., description="投诉文本，≤500 字")
    model: str = Field("complementnb", description="模型 key：complementnb / sgd_hinge / bert_nf4 / bert_int8")


@app.get("/", response_class=HTMLResponse)
def index():
    return render(REGISTRY.page_models())


@app.get("/models")
def models():
    return REGISTRY.page_models()


@app.post("/predict")
def predict(payload: PredictIn):
    text = (payload.text or "").strip()
    key = payload.model or "complementnb"
    if not text:
        return JSONResponse({"error": "文本不能为空"}, status_code=400)
    if len(text) > 500:
        return JSONResponse({"error": "文本请控制在 500 字以内"}, status_code=400)
    if key not in [m["key"] for m in REGISTRY.page_models()]:
        return JSONResponse({"error": f"未知模型：{key}"}, status_code=400)
    ok, reason = REGISTRY.status[key]
    if not ok:
        return JSONResponse(
            {"error": f"{REGISTRY.spec(key)['label']} 当前不可用：{reason}"}, status_code=503)
    try:
        return REGISTRY.predict_top5(key, text)
    except Exception as e:
        return JSONResponse({"error": f"预测出错：{e}"}, status_code=500)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
    print(f"页面： http://127.0.0.1:{port}  |  接口文档： http://127.0.0.1:{port}/docs")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
