# -*- coding: utf-8 -*-
"""5.10 结论的两张韦恩图。

图一（17_venn_cluster.png）：互渗簇定性韦恩 —— 电商平台 / 服饰鞋包 / 家居日用
  三个类两两类中心余弦均 >= 0.82（全 153 对均值 0.623），重叠区写的是该区域
  投诉文本的典型形态；重叠区的判定权在"主诉对象"，这是标注规范里的隐含规则。

图二（18_venn_rescue.png）：模型互补定量韦恩 —— 每类一个 venn2，两个集合分别
  是 LinearSVC 判对与 BERT 判对的样本集合，数字来自逐样本联合分布
  （resources/bert_svc_joint.json，由 Kaggle 内核 wubarry/textcls-bert-svc-joint
  在无泄漏测试集上算出，划分与 5.5 节完全一致）。圆外数字 = 两模型都错的样本
  数，即该类"真模糊"样本的下限口径（对这两个模型而言）。

输入：resources/bert_svc_joint.json
输出：resources/figures/17_venn_cluster.png、18_venn_rescue.png
"""
import os
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Circle
from matplotlib_venn import venn2

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False

DPI = 140
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "resources")
OUT = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)

INK = "#0f172a"
C_SVC = "#3b82f6"      # LinearSVC（蓝）
C_BERT = "#d946ef"     # BERT（洋红），与打榜图配色一致
C_E = "#f59e0b"        # 电商平台（橙）
C_F = "#10b981"        # 服饰鞋包（绿）
C_H = "#3b82f6"        # 家居日用（蓝）

# ---------------- 图一：互渗簇定性韦恩 ----------------
fig, ax = plt.subplots(figsize=(10.5, 7.9), facecolor="white", dpi=DPI)
ax.set_xlim(0, 10)
ax.set_ylim(-0.42, 8.35)
ax.set_axis_off()

R = 2.35
centers = {
    "电商平台": (3.6, 5.35),
    "服饰鞋包": (6.4, 5.35),
    "家居日用": (5.0, 3.2),
}
styles = {"电商平台": C_E, "服饰鞋包": C_F, "家居日用": C_H}
LABEL_POS = {"电商平台": (3.6, 5.35 + R + 0.28, "bottom"),
             "服饰鞋包": (6.4, 5.35 + R + 0.28, "bottom"),
             "家居日用": (5.0, 3.2 - R - 0.30, "top")}
for name, (x, y) in centers.items():
    ax.add_patch(Circle((x, y), R, facecolor=styles[name], alpha=0.16,
                        edgecolor=styles[name], lw=2.2))
    lx, ly, va = LABEL_POS[name]
    ax.text(lx, ly, name, ha="center", va=va,
            fontsize=15, fontweight="bold", color=styles[name])


def put(x, y, s, size=10.5, color=INK, weight="normal", ha="center"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color,
            fontweight=weight, linespacing=1.45)


# 三块"独有区"
put(2.35, 5.5, "平台自身行为\n保价 / 服务费 / 仲裁 / 不作为", 10.5)
put(7.65, 5.5, "商品本身质量\n无平台语境（鞋码 / 开胶）", 10.5)
put(5.0, 2.15, "日用品使用体验\n（不涉购买纠纷）", 10.5)
# 两两重叠区
put(5.0, 6.05, "同构反向标注区\n「拼多多 + 鞋 + 质量问题 + 平台介入」\n判定权：主诉对象是平台还是商品", 9.2, "#b45309", "bold")
put(4.05, 4.05, "电商买日用品的纠纷", 9.6)
put(5.95, 4.05, "鞋服家纺质感 / 尺码", 9.6)
# 中心
put(5.0, 5.0, "商品质量 × 电商平台语境", 9.6, "#6d28d9", "bold")

put(5.0, -0.22, "类中心余弦相似对：电商平台—服饰鞋包 0.849（全 153 对最大）、电商平台—家居日用 0.844、服饰鞋包—家居日用 0.823（均值 0.623）",
    9.2, "#64748b")
fig.suptitle("互渗簇：三个类的判据重叠在主诉对象，文本表层不携带",
             fontsize=13.5, fontweight="bold", color=INK, y=0.985)

fig.savefig(os.path.join(OUT, "17_venn_cluster.png"),
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("17_venn_cluster.png done")

# ---------------- 图二：模型互补定量韦恩 ----------------
joint = json.load(open(os.path.join(RES, "bert_svc_joint.json"), encoding="utf-8"))
CLASSES = ["电商平台", "影音娱乐", "家居日用", "共享出行"]
TITLE = {"电商平台": "瓶颈类 · BERT 主要赢在这",
         "影音娱乐": "并列最难 · BERT 唯一失手处",
         "家居日用": "互渗簇枢纽",
         "共享出行": "对照：重叠温和的类"}

fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.6), facecolor="white", dpi=DPI)
for ax, cls in zip(axes.flat, CLASSES):
    j = joint[cls]
    n = j["n"]
    v = venn2(subsets=(j["svc_only"], j["bert_only"], j["both"]),
              set_labels=("LinearSVC 判对", "BERT 判对"), ax=ax,
              set_colors=(C_SVC, C_BERT), alpha=0.45)

    def fmt(c):
        return f"{c}\n{c / n:.0%}"

    for lid, key in (("10", "svc_only"), ("01", "bert_only"), ("11", "both")):
        lab = v.get_label_by_id(lid)
        if lab:
            lab.set_text(fmt(joint[cls][key]))
            lab.set_fontsize(11)
            lab.set_fontweight("bold")
    for text in v.set_labels:
        if text:
            text.set_fontsize(11.5)
            text.set_fontweight("bold")
    ax.set_title(f"{cls}（n={n}）\n都错 {j['neither']} 条 = 两模型视角下的真模糊 · {TITLE[cls]}",
                 fontsize=11.5, color=INK, pad=10)

fig.suptitle("LinearSVC 与 BERT(epoch 59) 逐样本联合分布 · 无泄漏测试集 1,378 条 · "
             f"全集：LinearSVC {joint['_overall']['svc_acc']:.1%} vs BERT {joint['_overall']['bert_acc']:.1%}"
             f"（BERT 净多对 {joint['_overall']['bert_only'] - joint['_overall']['svc_only']} 条）",
             fontsize=13, fontweight="bold", color=INK, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(os.path.join(OUT, "18_venn_rescue.png"),
            bbox_inches="tight", facecolor="white")
plt.close(fig)
print("18_venn_rescue.png done")
for c in CLASSES:
    j = joint[c]
    print(f"{c:6s} n={j['n']:4d} 都对={j['both']:4d} 仅SVC={j['svc_only']:3d} "
          f"仅BERT={j['bert_only']:3d} 都错={j['neither']:3d}")
