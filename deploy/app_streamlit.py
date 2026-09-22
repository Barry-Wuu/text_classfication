# -*- coding: utf-8 -*-
"""Streamlit 部署（讲义三种部署方式里的最快落地版）。

启动：
  cd deploy && streamlit run app_streamlit.py
"""
import time

import streamlit as st

from model_registry import REGISTRY

st.set_page_config(page_title="投满分 · 消费者投诉文本分类", page_icon="🧭", layout="centered")
st.title("🧭 投满分 · 消费者投诉文本分类")
st.caption("选一个帕累托最优模型，输入投诉文本，看 Top5 类别与概率")

specs = REGISTRY.page_models()
options = {f"{m['label']}（{m['device']} · F1 {m['macro_f1']}）" + ("" if m["available"] else " · 不可用"): m
           for m in specs}
choice = st.selectbox("选择模型", list(options.keys()),
                      index=0 if specs[0]["available"] else
                      next(i for i, k in enumerate(options) if options[k]["available"]))
chosen = options[choice]
if not chosen["available"]:
    st.warning(f"{chosen['label']} 当前不可用：{chosen['reason']}")

text = st.text_area("投诉文本（≤500 字，Ctrl+Enter 快捷预测）", height=140,
                    placeholder="例如：在拼多多买的运动鞋开胶，商家拒绝退货，申请平台介入无果")
col1, col2 = st.columns([1, 1])
run = col1.button("预测", type="primary", use_container_width=True)
if col2.button("清空", use_container_width=True):
    st.rerun()

if run:
    if not text.strip():
        st.error("请输入投诉文本")
    elif len(text.strip()) > 500:
        st.error("文本请控制在 500 字以内")
    elif not chosen["available"]:
        st.error(f"{chosen['label']} 不可用：{chosen['reason']}")
    else:
        with st.spinner("推理中…"):
            t0 = time.perf_counter()
            res = REGISTRY.predict_top5(chosen["key"], text.strip())
            rtt = (time.perf_counter() - t0) * 1000
        st.subheader("Top5 预测")
        for i, it in enumerate(res["top5"]):
            c1, c2, c3 = st.columns([1.2, 4, 1])
            c1.markdown(f"**{it['label']}**" if i == 0 else it["label"])
            c2.progress(min(it["prob"], 1.0))
            c3.markdown(f"{it['prob'] * 100:.2f}%")
        st.caption(f"模型：{res['model_label']}（{res['device']}） · "
                   f"服务端 {res['ms']} ms · 端到端 {rtt:.0f} ms")

with st.expander("本机模型可用性"):
    st.json(specs)
