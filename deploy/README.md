# 模型部署（Flask / Streamlit / FastAPI）

三种方式共用 `model_registry.py`，页面统一支持**三个帕累托最优模型任选**、输出 **Top5 类别 + 概率**并显示**响应时间**。

## 快速开始

```bash
cd deploy

# ① Flask 主页面（改造自 D:\B++\投满分.ipynb 的 Flask 单元）
python app_flask.py 8800          # http://127.0.0.1:8800

# ② Streamlit（最快跑起来的交互页）
streamlit run app_streamlit.py

# ③ FastAPI（服务化，自带 /docs）
python app_fastapi.py 8801        # http://127.0.0.1:8801/docs
```

演示链接（打开即预填并自动预测）：

```
http://127.0.0.1:8800/?model=sgd_hinge&demo=在拼多多买的运动鞋开胶，商家拒绝退货，申请平台介入无果&auto=1
```

## 模型与可用性

| key | 模型 | 设备 | macro-F1 | 体积 | 依赖 |
| --- | --- | --- | ---: | ---: | --- |
| `complementnb` | ComplementNB | CPU | 0.8068 | 3.5 MB | 随仓库提供 |
| `sgd_hinge` | SGD（hinge） | CPU | 0.8138 | 1.5 MB | 随仓库提供 |
| `bert_nf4` | BERT-NF4 4bit 量化 | GPU | 0.8590 | 108.2 MB | CUDA + bitsandbytes，权重需下载 |
| `bert_int8` | BERT-INT8 动态量化 | CPU | 0.8518 | 145.6 MB | 权重需下载 |

量化权重下载（GitHub Release）：

- `bert_nf4_state.pt` — https://github.com/Barry-Wuu/text_classfication/releases/download/deploy-models/bert_nf4_state.pt
- `bert_int8_state.pt` — https://github.com/Barry-Wuu/text_classfication/releases/download/deploy-models/bert_int8_state.pt
- `deploy_models_meta.json`（加载说明与指标）— 同 release 目录

下载后放到 `deploy/models/`（或设 `DEPLOY_MODEL_DIR` 指向所在目录）。BERT 基座默认 `D:\models\bert-base-chinese`，可用 `BASE_BERT` 覆盖。

## 重新训练两个非 BERT 模型

```bash
cd deploy && python train_export_models.py
```

与报告 5.5 节完全同协议（同一无泄漏划分、同一 TF-IDF 配置），训练完会打印回测指标，
写入 `models/metrics.json`；实测 ComplementNB 0.8128 / SGD(hinge) 0.8157（准确率），与报告逐位一致。

## 目录

```
deploy/
├── model_registry.py         模型注册表：Top5 接口 + 可用性探测（不预加载，省内存）
├── page_html.py              页面模板（改造自 投满分.ipynb 的 Flask 内嵌页）
├── app_flask.py              Flask 版
├── app_streamlit.py          Streamlit 版
├── app_fastapi.py            FastAPI 版
├── jieba_cut.py              模块级分词函数（joblib 序列化按引用记录）
├── train_export_models.py    训练 + 导出两个非 BERT 模型
└── models/                   模型文件（小模型入库；量化权重自行下载）
```

## 两个实现细节

1. **分词必须做成模块级函数**（`jieba_cut.cut`）。直接把 `jieba.lcut` 交给 `TfidfVectorizer`
   会导致 `joblib.dump` 报 `TypeError: cannot pickle '_thread.RLock' object`——
   `jieba.lcut` 是惰性加载的 `Tokenizer` 实例上的绑定方法，实例里带锁。
2. **概率口径**：ComplementNB 用 `predict_proba`（偏平，Top1 常 10%～20%）；
   SGD（hinge）没有 `predict_proba`，用 OvR `decision_function` 经 softmax 近似（排序可用、绝对值不可当真）；
   BERT 系列是正常 softmax。页面与 README 都标注了这一点。
