# 标签表与张量策略

本目录记录「中文消费投诉文本分类」数据集的标签编码方案与文本转张量策略。

## 一、标签表

数据 `labeled.csv`（30000 行）三列：`text`（特征）、`category`（中文类名）、`label`（0~17 整数标签）。

类名到整数 id 的映射按 **字典序** 固定（可复现），同时导出两种格式：

- `resources/class.txt`：一行一个类名，**行号即 label id**（参照 `TMFCode/01-data/class.txt` 格式）
- `resources/label_map.csv`：`id,name,count` 三列，含各类样本量

| id | 类名 | 样本量 |
|---:|---|---:|
| 0 | 共享出行 | 1768 |
| 1 | 医疗健康 | 1382 |
| 2 | 婚恋交友 | 1297 |
| 3 | 家居日用 | 1809 |
| 4 | 影音娱乐 | 1813 |
| 5 | 房产家装 | 1749 |
| 6 | 教育 | 1798 |
| 7 | 数码3C | 1775 |
| 8 | 旅游出行 | 1784 |
| 9 | 服饰鞋包 | 1821 |
| 10 | 本地生活 | 1848 |
| 11 | 母婴食品 | 416 |
| 12 | 汽车 | 1796 |
| 13 | 游戏 | 1775 |
| 14 | 物流快递 | 1766 |
| 15 | 电商平台 | 1816 |
| 16 | 通讯运营商 | 1771 |
| 17 | 金融支付 | 1816 |

生成脚本：`resources/build_labels.py`。校验：id↔类名双向一致、18 类标签唯一。

## 二、文本转张量策略

截断长度统一取 **`max_length=200`**（此前实测 P95≈201，>200 仅截极少数样本、几乎无损）。

### 方案 A：BERT tokenizer

用 `D:\models\bert-base-chinese`（词表 21128），逐字切分，转固定长度张量：

- `resources/tensors_bert.npz`：`input_ids` / `attention_mask` / `token_type_ids`（int32，`[30000,200]`）+ `labels`
- `resources/bert_token_lengths.npy`：截断前真实长度

实测：编码 30000 条约 4 秒；真实长度（含 CLS/SEP）P50=156、P95=184、P99=191、max=206，**超 200 被截断的仅 14 条（0.05%）**；npz 体积 5.1 MB。

脚本：`resources/to_tensor_bert.py`。

### 方案 B：jieba + 词频自训词表

jieba 分词 → 词频统计 → 构建词表（`<pad>=0`、`<unk>=1`，过滤纯标点，`min_freq>=2`）→ 整数序列 pad/truncate 到 200：

- `resources/vocab.txt`：一行一词，**行号即 id**
- `resources/tensors_tfidf.npz`：`text_ids [30000,200]` + `labels` + `lengths`
- `resources/tfidf_matrix.npz` / `tfidf_vectorizer.pkl`：TF-IDF 稀疏矩阵（对照）

实测：jieba 分词 30000 条约 19 秒；词表 28897；分词后序列长度 P50=93、P95=114、max=150（200 长度充裕，**截断 0 条**）；Top20 高频词为「我/的/了/在/月/不/年/平台/本人/元/日/9/客服/…」。脚本：`resources/to_tensor_tfidf.py`。

## 三、两条策略效果对比

同一 8:2 分层划分（seed=42）、同一逻辑回归分类头，只替换"文本如何变张量"：

| 策略 | 特征维度 | macro-F1 | 说明 |
|---|---:|---:|---|
| A. BERT [CLS] 句向量 | 768 | 见实测 | 不微调，纯特征提取 |
| B. 词频 EmbeddingBag | 256 | 见实测 | 随机嵌入 + 均值池化 |
| C. TF-IDF 稀疏 | 28806 | 见实测 | 传统基线 |

脚本：`resources/compare_tensors.py`。结论与推荐见运行输出与本报告末节。
