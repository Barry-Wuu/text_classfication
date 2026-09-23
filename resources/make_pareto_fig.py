# -*- coding: utf-8 -*-
"""打榜风格的帕累托图：横轴=耗时(秒,对数)，纵轴=macro-F1，原点在左下角，
并绘制帕累托最优边界（越靠左上越好：同样耗时下准确率最高、同样准确率下最快）。

标签排版：方法名与数据点分离，用细引线相连；每个标签在一组候选方位里贪心挑
"冲突代价最小"的位置（帕累托前沿优先占位），并回避数据点圆盘与其它标签。
最后用"隐藏标签前后做像素差分"做地面真值自检，确认没有东西画到数据点上。

四个必须记住的坑：
  1) 坐标轴（set_xscale / margins）与 tight_layout 必须先定型，再做标签排版，
     否则算出的像素坐标与最终画布不一致。
  2) 画布 dpi 必须与导出 dpi 一致（这里统一用 DPI 常量），否则显示坐标与 PNG
     像素差 dpi/100 倍，像素级自检会落在错误位置。
  3) annotate 的引线是从"文字框中心"（relpos 默认 (0.5, 0.5)）画到数据点的，
     不是从文字边缘锚点出发；按锚点建模会低估引线长度、漏判穿孔。
  4) Annotation.get_window_extent() 返回的是"文字 + 引线"的联合外框，不能当
     文字盒子用，要另建同字号临时 Text 来测。

输出：resources/figures/10_pareto.png
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False
rcParams["font.size"] = 11

DPI = 140          # 画布与导出统一用这个 dpi（像素自检按它算）
FIG_W, FIG_H = 11.5, 7.2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "resources")
OUT = os.path.join(RES, "figures")
os.makedirs(OUT, exist_ok=True)

d = json.load(open(os.path.join(RES, "tfidf_clf_result.json"), encoding="utf-8"))
rows = d["rows"]


def cost(r):
    """统一到"单条推理耗时"口径，让本地模型与 API 模型可比。

    本地模型：训练耗时 + 推理耗时（fit_s + infer_s）。
    大模型 API：只有一次全测试集推理的墙钟时间 infer_s（无训练概念）。
    为公平起见，横轴统一取"完成整轮评测的墙钟耗时"：
      本地 = fit_s + infer_s； API = infer_s。
    """
    fit = r.get("fit_s") or 0.0
    inf = r.get("infer_s") or 0.0
    return fit + inf


def color(n):
    """按阵营着色：BERT 系（含其压缩产物）一色，传统机器学习及其组合一色，
    第三方大模型 API 一色。"""
    if "LLM" in n or "Jev" in n:
        return "#ef4444"          # 第三方大模型 API 红
    if "BERT" in n or "蒸馏" in n:
        return "#db2777"          # BERT 系（微调 / 量化 / 剪枝 / 蒸馏产物）洋红
    return "#3b82f6"              # 传统机器学习及其组合 蓝


pts = [dict(name=r["name"], t=cost(r), f1=r["macro_f1"]) for r in rows]

# ---- 帕累托最优：耗时越小越好、F1 越大越好 ----
# 一条点 p 被支配 <=> 存在 q 使 q.t <= p.t 且 q.f1 >= p.f1（至少一项严格更好）
pareto = []
for p in pts:
    dominated = False
    for q in pts:
        if q is p:
            continue
        if q["t"] <= p["t"] and q["f1"] >= p["f1"] and (q["t"] < p["t"] or q["f1"] > p["f1"]):
            dominated = True
            break
    if not dominated:
        pareto.append(p)
pareto.sort(key=lambda x: x["t"])


def is_pareto(p):
    return any(p is q for q in pareto)


BG = "#ffffff"
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=BG, dpi=DPI)
ax.set_facecolor(BG)

# 数据点
for p in pts:
    on = is_pareto(p)
    ax.scatter(p["t"], p["f1"], s=190 if on else 130,
               color=color(p["name"]), edgecolor="#1e293b",
               linewidth=1.3 if on else 0.6, zorder=3)

# 帕累托前沿阶梯线（从左下到右上，先横后竖）
if pareto:
    xs, ys = [], []
    for i, p in enumerate(pareto):
        if i == 0:
            xs.append(p["t"]); ys.append(p["f1"])
        else:
            xs.append(p["t"]); ys.append(pareto[i - 1]["f1"])  # 水平段
            xs.append(p["t"]); ys.append(p["f1"])              # 竖直段
    ax.plot(xs, ys, color="#dc2626", ls="--", lw=1.8, alpha=0.85,
            zorder=2, label="帕累托最优边界")

# ---- 坐标轴先定型：对数轴、标题、参考线、外边距 ----
ax.set_xscale("log")
ax.set_xlabel("耗时（秒，对数轴；本地 = 训练+推理，大模型 API = 全量推理）",
              fontsize=11)
ax.set_ylabel("macro-F1（无泄漏划分）", fontsize=11)
ax.set_title("打榜视角 · 帕累托最优：耗时 vs 准确率（越靠左上越优）",
             fontsize=14.5, fontweight="bold")

# 原点方向提示（左下角）：优方向是 耗时更小(左) + F1 更大(上) = 左上(↖)
ax.annotate("", xy=(0.015, 0.14), xytext=(0.085, 0.025),
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.4))
ax.text(0.09, 0.045, "越靠左上越好\n（更快 · 更准）", transform=ax.transAxes,
        fontsize=9.5, color="#64748b", ha="left", va="bottom")

# 参考线：基线 F1
base_f1 = max(r["macro_f1"] for r in rows)
ax.axhline(base_f1, color="#3b82f6", ls=":", lw=1.1, alpha=0.55)
ax.text(ax.get_xlim()[0], base_f1, f" 最高 macro-F1 = {base_f1:.4f}",
        color="#2563eb", fontsize=9, va="bottom")

ax.grid(True, which="both", ls=":", alpha=0.35)
ax.set_axisbelow(True)
ax.margins(x=0.08, y=0.12)

fig.text(0.5, 0.012,
         "洋红=BERT 系（微调及其压缩产物：量化 / 剪枝 / 蒸馏）　蓝=传统机器学习及其组合（线性 / 贝叶斯 / 树 / kNN / fastText / 集成）　红=第三方大模型 API　"
         "　红点+虚线=帕累托最优前沿（不被任何方法支配）",
         ha="center", fontsize=9, color="#64748b")
plt.tight_layout(rect=[0, 0.03, 1, 1])   # 必须先于标签排版：它会改变 axes 位置


# ============ 标签排版：名称与点分离，引线相连，候选方位贪心选优 ============
def short(n):
    return n.replace("LLM:", "").replace("(cosine)", "")


def to_px(xy):
    """数据坐标 -> 显示像素（始终取当前 transform）。"""
    return ax.transData.transform(xy)


SHRINK_A, SHRINK_B = 3.0, 11.0   # 引线两端内缩（点），与 annotate 设置保持一致
anns = []
for p in sorted(pts, key=lambda x: x["t"]):
    on = is_pareto(p)
    t = ax.annotate(short(p["name"]), (p["t"], p["f1"]),
                    textcoords="offset points", xytext=(0, 0),
                    fontsize=9.8 if on else 8.5,
                    fontweight="bold" if on else "normal",
                    color="#111827" if on else "#475569", zorder=6,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.9,
                                    color=color(p["name"]), alpha=0.8,
                                    shrinkA=SHRINK_A, shrinkB=SHRINK_B))
    anns.append((p, t, on))

fig.canvas.draw()
_rend = fig.canvas.get_renderer()
_axbb = ax.get_window_extent(renderer=_rend)
_PT = DPI / 72.0                        # 1 point -> 像素

# 文字真实外框偏移：Annotation.get_window_extent() 会把引线也算进去，
# 不能拿来当文字盒子，这里用同字号的临时 Text 精确量"锚点 -> 文字外框"。
PROBE = {}
_tmp = []
for p, t, on in anns:
    for ha in ("left", "right"):
        tx = ax.text(p["t"], p["f1"], short(p["name"]),
                     fontsize=9.8 if on else 8.5,
                     fontweight="bold" if on else "normal",
                     ha=ha, va="center", alpha=0.0)
        _tmp.append((id(t), ha, tx, p))
fig.canvas.draw()
for k, ha, tx, p in _tmp:
    bb = tx.get_window_extent(renderer=_rend)
    P = to_px((p["t"], p["f1"]))
    dxr = (bb.x0 - P[0]) if ha == "left" else (bb.x1 - P[0])
    PROBE[(k, ha)] = (bb.width, bb.height, dxr,
                      (bb.y0 + bb.y1) / 2 - P[1])
    tx.remove()
fig.canvas.draw()

# 数据点障碍盒：半径由散点面积(s, pt^2)反算；s=190/130 在 140dpi 下直径约 30/26 px
OBST = []
for p in pts:
    _r = np.sqrt((190.0 if is_pareto(p) else 130.0) / np.pi) * _PT
    _c = to_px((p["t"], p["f1"]))
    OBST.append((_c[0], _c[1], 2 * _r + 2.0, 2 * _r + 2.0, p))

PAD = 3.0
ANGLES = np.arange(0.0, 360.0, 15.0)
RADII = (26.0, 40.0, 56.0, 74.0, 94.0, 118.0, 146.0)          # 引线长度候选（点）
_xbar = float(np.mean([to_px((p["t"], p["f1"]))[0] for p in pts]))


def pen(cx, cy, w, h, ox, oy, ow, oh):
    """带内边距的穿透代价（截断，避免大标签把权重吃光）。"""
    dx = (w + ow) / 2 + PAD - abs(cx - ox)
    dy = (h + oh) / 2 + PAD - abs(cy - oy)
    if dx <= 0 or dy <= 0:
        return 0.0
    return min(dx, 25.0) * min(dy, 20.0)


def seg_hits_disk(A, B, c, r):
    """线段 A->B 是否穿过圆心 c 半径 r 的圆盘。"""
    AB = B - A
    t = float(np.dot(c - A, AB)) / max(float(np.dot(AB, AB)), 1e-9)
    t = min(1.0, max(0.0, t))
    return float(np.hypot(*(A + t * AB - c))) < r


# 帕累托前沿（加粗、最重要）先占位，其余按标签宽度从大到小
seq = sorted(anns, key=lambda a: (0 if a[2] else 1,
                                  -PROBE[(id(a[1]), "left")][0]))
placed, relax = [], []
for p, t, on in seq:
    P = to_px((p["t"], p["f1"]))
    cands = []
    for rad in RADII:
        for ang in ANGLES:
            ux, uy = float(np.cos(np.radians(ang))), float(np.sin(np.radians(ang)))
            if abs(ux) < 0.35:                  # 不用正上/正下，避免标签竖成一列
                continue
            ha = "left" if ux >= 0 else "right"
            w, h, dxr, dyr = PROBE[(id(t), ha)]
            dx, dy = ux * rad, uy * rad
            ax_, ay_ = P[0] + dx * _PT, P[1] + dy * _PT
            x0 = (ax_ + dxr) if ha == "left" else (ax_ + dxr - w)
            cx, cy = x0 + w / 2, ay_ + dyr
            if (cx - w / 2 < _axbb.x0 + 3 or cx + w / 2 > _axbb.x1 - 3 or
                    cy - h / 2 < _axbb.y0 + 3 or cy + h / 2 > _axbb.y1 - 3):
                continue                        # 出界直接否决
            obst_pen = 0.0
            for (ox, oy, ow, oh, _op) in OBST:
                obst_pen += pen(cx, cy, w, h, ox, oy, ow, oh)
            # 引线穿越数据点圆盘：注意 annotate 的引线是从"文字框中心"
            # （relpos 默认 (0.5,0.5)）画到数据点的，不是从文字边缘锚点出发，
            # 用锚点建模会低估长度、漏判穿孔。末端还要按 shrinkB 截掉。
            _CA = np.array([cx, cy])
            _d = P - _CA
            _L = float(np.hypot(*_d))
            _u = _d / _L if _L > 1e-9 else np.array([1.0, 0.0])
            _A2 = _CA + _u * (SHRINK_A * _PT)
            _B2 = P - _u * (SHRINK_B * _PT)
            cross = 0
            for (ox, oy, ow, oh, _op) in OBST:
                if seg_hits_disk(_A2, _B2, np.array([ox, oy]), ow / 2 + 1.5):
                    cross += 1
            cost = 1.0 * obst_pen + 60.0 * cross
            for (pcx, pcy, pw, ph) in placed:
                c2 = 3.0 * pen(cx, cy, w, h, pcx, pcy, pw, ph)
                if abs(cy - pcy) < 0.5 * min(h, ph):
                    c2 *= 2.4                   # 同一水平线：额外要求整行间隙
                cost += c2
            if (ux > 0) == (P[0] < _xbar):      # 引线别朝数据云内部扎
                cost += 25.0
            cost += 0.5 * rad                   # 引线越短越好
            cands.append((cost, obst_pen, cross, dx, dy, ha, cx, cy, w, h))
    # 硬约束优先：先只在"不压点且引线不穿点"的候选里挑，全不满足才退让
    strict = [c for c in cands if c[1] <= 4.0 and c[2] == 0]
    best = min(strict or cands, key=lambda c: c[0]) if cands else None
    if best is None:                            # 兜底
        _w0, _h0, _dxr0, _dyr0 = PROBE[(id(t), "left")]
        best = (1e9, 0.0, 0, 30.0, 0.0, "left",
                P[0] + 30.0 * _PT + _dxr0 + _w0 / 2, P[1] + _dyr0, _w0, _h0)
    _c, _op, _cr, dx, dy, ha, cx, cy, w, h = best
    if _op > 4.0 or _cr > 0:                    # 无严格可行解，只能退让
        relax.append((p["name"], round(_op, 1), _cr))
    t.set_position((dx, dy))
    t.set_ha(ha)
    placed.append((cx, cy, w, h))

# ---- 复核一：几何口径（标签之间 / 标签与数据点）----
bad = 0
for i in range(len(placed)):
    for j in range(i + 1, len(placed)):
        cx, cy, w, h = placed[i]
        pcx, pcy, pw, ph = placed[j]
        if (abs(cx - pcx) < (w + pw) / 2 + PAD and
                abs(cy - pcy) < (h + ph) / 2 + PAD):
            bad += 1
hit = 0
for (cx, cy, w, h) in placed:
    for (ox, oy, ow, oh, _op) in OBST:
        if (abs(cx - ox) < (w + ow) / 2 and abs(cy - oy) < (h + oh) / 2):
            hit += 1

# ---- 复核二：像素地面真值（渲染差分：标签/引线是否盖住数据点）----
# 按文字颜色取墨迹会误伤标记描边的抗锯齿混色，改为"隐藏标签前后做像素差分"：
# 圆内部若出现差异像素，说明确有东西画到了点上。
fig.canvas.draw()
buf_on = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int).copy()
for _, t, _ in anns:
    t.set_visible(False)
fig.canvas.draw()
buf_off = np.asarray(fig.canvas.buffer_rgba())[..., :3].astype(int).copy()
for _, t, _ in anns:
    t.set_visible(True)

diff = np.abs(buf_on - buf_off).sum(axis=2) > 20
_H, _W = diff.shape[:2]
assert (_H, _W) == (int(round(FIG_H * DPI)), int(round(FIG_W * DPI))), \
    f"画布尺寸 {(_H, _W)} 与导出不一致，像素坐标会错位"
_yy, _xx = np.mgrid[0:_H, 0:_W]
covered = []
for p in pts:
    cx, cy = to_px((p["t"], p["f1"]))
    r = np.sqrt((190.0 if is_pareto(p) else 130.0) / np.pi) * _PT
    row = _H - 1 - cy                       # buffer 行号自顶向下
    disk = (_xx - cx) ** 2 + (_yy - row) ** 2 <= max(r - 1.5, 1.0) ** 2
    n = int((diff & disk).sum())
    if n > 0:
        yy2, xx2 = np.nonzero(diff & disk)
        covered.append((p["name"], n,
                        f"x:{xx2.min()}-{xx2.max()} y(row):{yy2.min()}-{yy2.max()}"))

print("布局退让（无严格可行解）:", relax if relax else "无")
print("几何复核 · 标签互叠:", bad, "| 标签压点:", hit)
print("像素复核 · 被盖住的数据点:", covered if covered else "无")
print("帕累托前沿成员:", [q["name"] for q in pareto])

p_out = os.path.join(OUT, "10_pareto.png")
fig.savefig(p_out, dpi=DPI, facecolor=BG)
print("已写:", p_out)
