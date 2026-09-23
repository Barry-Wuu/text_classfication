# Kaggle 内核源码

本目录是项目全部 Kaggle 内核的源码。这些实验放在 Kaggle 而不是本机跑，原因有两个：一是需要 GPU（BERT 微调、NF4 量化、剪枝、蒸馏），二是机房上行快——390MB 的权重从 Kaggle 直传 GitHub Release 只需十几秒，本机则要几十分钟。

内核产出落在 `resources/` 下，与本地脚本共用同一套无泄漏协议（`GroupShuffleSplit(test_size=0.2, random_state=42)` 按源 id 划分），因此两边数字可以直接对齐。

## 一、复现顺序

| 步骤 | 目录 | 内核 id | 运行前需要 |
| ---: | --- | --- | --- |
| 1 | `cmp_int8/` | `wubarry/textcls-cmp-int8` | 无（CPU 内核，自带教师 CPU 基准） |
| 2 | `cmp_nf4/` | `wubarry/textcls-cmp-nf4` | GPU |
| 3 | `cmp_prune/` | `wubarry/textcls-cmp-prune` | GPU |
| 4 | `cmp_kd2/` | `wubarry/textcls-cmp-kd2` | GPU（LSTM 128 维，8 轮） |
| 5 | `cmp_kd3/` | `wubarry/textcls-cmp-kd3` | GPU（LSTM 256 维，20 轮，最终采用） |
| 6 | `export_models/` | `wubarry/textcls-deploy-export` | GPU + `GH_TOKEN` |
| 7 | `bert_cm/` | `wubarry/textcls-bert-cm` | 无（CPU 推理即可） |
| 8 | `bert_joint/` | `wubarry/textcls-bert-joint2` | 无 |
| 9 | `migrate_release/` | `wubarry/textcls-release-migrate` | 无 + `GH_TOKEN` |
| 10 | `github_bert_push/` | `wubarry/textcls-bert-gh59` | 无 + `GH_TOKEN` |

推送与拉取：

```bash
kaggle kernels push -p kaggle/cmp_kd3          # 推送（会自动跑）
kaggle kernels status wubarry/textcls-cmp-kd3  # 查状态
kaggle kernels output wubarry/textcls-cmp-kd3 -p out/kd3   # 拉产物
```

## 二、GH_TOKEN 的配置

`export_models/`、`migrate_release/`、`github_bert_push/` 三个内核要把产物发布到 GitHub Release，需要令牌。源码里**不保存任何令牌**，统一从环境变量读取：

```python
TOKEN = os.environ.get('GH_TOKEN', '').strip()
```

在 Kaggle 内核里通过 **Add-ons → Secrets** 添加名为 `GH_TOKEN` 的密钥即可自动注入；本地跑则自行 `export GH_TOKEN=...`。

令牌需要 `repo` 权限（`migrate_release` 还需要读取源仓库的 Release）；发布到 Release 用不上 `delete_repo`。

## 三、各内核做什么

**压缩实验**（README 5.12 节）

| 目录 | 方式 | 结果（教师 F1 0.8660） |
| --- | --- | --- |
| `cmp_int8/` | INT8 动态量化，CPU 推理 | 145.6 MB，F1 0.8518 |
| `cmp_nf4/` | NF4 4bit 量化（bitsandbytes） | 108.1 MB，GPU 2.09 ms/样本，F1 0.8590 |
| `cmp_prune/` | 全局 L1 非结构化剪枝 30% 与 50% | 30% F1 0.8595 但体积不变；50% 崩到 0.7704 |
| `cmp_kd/` | 软标签蒸馏 v1，沿用 BERT 抽样协议 | 仅 0.5131（反面实验：小模型必须全量训练） |
| `cmp_kd2/` | 软标签蒸馏 v2，LSTM 128 维 8 轮 | 11.3 MB，F1 0.6923 |
| `cmp_kd3/` | 软标签蒸馏 v3，LSTM 256 维 20 轮 | 24.7 MB，CPU 2.15 ms/样本，F1 0.8083 |

三个蒸馏内核共用同一套配方：`loss = (1 - alpha) * 硬标签交叉熵 + alpha * T^2 * KL(软标签)`，取 `T = 2`、`alpha = 0.7`；教师软标签预缓存，避免每轮重复前向。

**结果复算**

| 目录 | 用途 | 产出 |
| --- | --- | --- |
| `bert_cm/` | BERT 在同一测试集上的混淆矩阵 | `resources/bert_confusion.json`（acc 0.8628 / macro-F1 0.8660） |
| `bert_joint/` | LinearSVC 与 BERT 的逐样本联合分布 | `resources/bert_svc_joint.json`（5.10 节韦恩图的数据源） |

**发布与搬运**

| 目录 | 用途 |
| --- | --- |
| `export_models/` | 把 NF4 与 INT8 量化权重直传 Release `deploy-models`（云端下载再上传，不经过本机） |
| `migrate_release/` | 跨仓库搬运 Release 附件（逐文件处理、传完即删，适配 Kaggle 输出盘容量） |
| `github_bert_push/` | BERT 打榜权重直传 Release `bert-weights` |

**生成器与弃用版本**

| 文件 | 说明 |
| --- | --- |
| `gen_kernels.py` | 由实验参数表批量生成上述压缩内核（改一处配置即可重出全部内核） |
| `deprecated/compress_bert_student.py` | 弃用：用大 BERT 蒸馏小 BERT，耗时且收益不划算 |
| `deprecated/compress_lstm_v2_broken.py` | 弃用：早期 LSTM 学生版本，`evaluate` 未取 `.logits` 导致报错 |

## 四、注意事项

- 权重不随仓库分发，下载地址见 README 7.3 节（GitHub Release）。
- 内核脚本里的数据与权重地址指向本仓库 `main` 分支与 Release，不依赖任何外部仓库。
- 所有脚本都做了凭证脱敏，仓库内不含任何明文令牌。
