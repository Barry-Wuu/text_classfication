# -*- coding: utf-8 -*-
"""页面模板：由 D:\\B++\\投满分.ipynb 的 Flask 内嵌页面改造而来。

保留原版的视觉语言（渐变背景 + 毛玻璃卡片 + 加载动画 + 示例标签 + 回车触发），
在此基础上增加三处功能：
  1) 模型选择：ComplementNB / SGD（hinge）/ BERT-NF4 4bit 三个帕累托最优模型；
     GPU 不可用时该选项置灰并给出原因（不静默换模型）；
  2) 结果区展示 Top5 类别 + 概率条 + 百分比；
  3) 记录响应时间（服务端推理耗时 + 往返耗时），并带清空按钮。
"""
import json

PAGE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>投满分 · 消费者投诉文本分类</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #6a3093 100%);
    min-height: 100vh; display: flex; justify-content: center; align-items: center;
    padding: 24px;
  }
  .card {
    background: rgba(255,255,255,0.96); backdrop-filter: blur(10px);
    border-radius: 24px; box-shadow: 0 20px 60px rgba(0,0,0,0.3);
    padding: 36px; max-width: 760px; width: 100%;
    animation: fadeIn 0.6s ease;
  }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
  h1 { font-size: 25px; font-weight: 600; color: #1e3c72; text-align: center; margin-bottom: 6px; }
  .sub { text-align: center; color: #8899aa; font-size: 13.5px; margin-bottom: 22px; }
  .models { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
  .model-opt {
    flex: 1 1 30%; min-width: 150px; padding: 10px 12px; border: 2px solid #e0e6ed;
    border-radius: 12px; cursor: pointer; transition: all 0.2s; background: #fff;
    font-size: 13px; line-height: 1.5; color: #33475b;
  }
  .model-opt:hover { border-color: #9db4d8; }
  .model-opt.active { border-color: #2a5298; background: #eef3fb; box-shadow: 0 0 0 4px rgba(42,82,152,0.12); }
  .model-opt.disabled { opacity: 0.55; cursor: not-allowed; background: #f7f8fa; }
  .model-opt .name { font-weight: 600; color: #1e3c72; font-size: 13.5px; }
  .model-opt .meta { color: #8899aa; font-size: 12px; }
  .badge { display: inline-block; padding: 1px 7px; border-radius: 8px; font-size: 11px;
           background: #e8f0fe; color: #2a5298; margin-left: 4px; }
  .badge.gpu { background: #fdeef7; color: #a21caf; }
  .input-area { display: flex; gap: 10px; align-items: flex-start; }
  textarea {
    flex: 1; padding: 14px 16px; font-size: 14.5px; line-height: 1.6;
    border: 2px solid #e0e6ed; border-radius: 14px; outline: none; resize: vertical;
    min-height: 96px; font-family: inherit; transition: border 0.2s, box-shadow 0.2s;
  }
  textarea:focus { border-color: #2a5298; box-shadow: 0 0 0 4px rgba(42,82,152,0.15); }
  .btns { display: flex; flex-direction: column; gap: 8px; }
  button {
    padding: 12px 24px; font-size: 14.5px; font-weight: 600; color: white;
    background: linear-gradient(135deg, #2a5298, #6a3093); border: none;
    border-radius: 14px; cursor: pointer; font-family: inherit;
    transition: transform 0.15s, box-shadow 0.2s; white-space: nowrap;
  }
  button.ghost { background: #eef1f6; color: #4a5b70; font-weight: 500; }
  button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(42,82,152,0.35); }
  button:active { transform: translateY(0); }
  button:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
  .counter { text-align: right; color: #b3bece; font-size: 12px; margin-top: 6px; }
  .result { margin-top: 20px; padding: 20px 22px; border-radius: 16px; background: #f7f9fc;
            opacity: 0; transform: translateY(10px); transition: opacity .4s, transform .4s; }
  .result.show { opacity: 1; transform: translateY(0); }
  .result .label { font-size: 12.5px; color: #8899aa; letter-spacing: 1px; margin-bottom: 12px; }
  .row { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
  .row .top1 .cat { font-weight: 700; }
  .cat { width: 84px; font-size: 14px; color: #1e3c72; }
  .bar-wrap { flex: 1; height: 22px; background: #e9eef6; border-radius: 11px; overflow: hidden; }
  .bar { height: 100%; border-radius: 11px; background: linear-gradient(90deg, #2a5298, #6a3093);
         transition: width .5s ease; }
  .pct { width: 62px; text-align: right; font-size: 13px; color: #4a5b70; font-variant-numeric: tabular-nums; }
  .foot { margin-top: 14px; padding-top: 12px; border-top: 1px dashed #dde3ea;
          font-size: 12.5px; color: #8899aa; display: flex; justify-content: space-between; }
  .examples { margin-top: 18px; border-top: 1px dashed #dde3ea; padding-top: 14px; }
  .examples .title { font-size: 12.5px; color: #8899aa; margin-bottom: 8px; }
  .example-tag { display: inline-block; padding: 5px 12px; margin: 3px; font-size: 12.5px;
                 color: #2a5298; background: #eef3fb; border-radius: 20px; cursor: pointer; }
  .example-tag:hover { background: #2a5298; color: #fff; }
  .error { color: #d9534f; font-size: 14px; }
  .loading { display: inline-block; width: 16px; height: 16px; border: 3px solid rgba(255,255,255,0.5);
             border-top-color: #fff; border-radius: 50%; animation: spin .8s linear infinite; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="card">
  <h1>🧭 投满分 · 消费者投诉文本分类</h1>
  <p class="sub">选一个帕累托最优模型，输入投诉文本，看 Top5 类别与概率</p>

  <div class="models" id="modelBox"></div>

  <div class="input-area">
    <textarea id="textInput" placeholder="例如：在拼多多买的运动鞋收到后发现开胶，联系商家要求退货被拒，申请平台介入也没有结果"></textarea>
    <div class="btns">
      <button id="predictBtn" onclick="doPredict()">预测</button>
      <button class="ghost" onclick="clearAll()">清空</button>
    </div>
  </div>
  <div class="counter"><span id="charCount">0</span> / 500</div>

  <div class="result" id="resultBox">
    <div class="label">Top5 预测</div>
    <div id="rowsBox"></div>
    <div class="foot">
      <span id="modelInfo">—</span>
      <span id="timeInfo">—</span>
    </div>
  </div>

  <div class="examples">
    <div class="title">试试这些例子：</div>
    <span class="example-tag" onclick="fillExample(this)">在拼多多买的运动鞋开胶，商家拒绝退货，申请平台介入无果</span>
    <span class="example-tag" onclick="fillExample(this)">买的视频网站年卡还没到期就停了，客服一直不回消息</span>
    <span class="example-tag" onclick="fillExample(this)">网约车司机绕路还拒载，平台只退了优惠券</span>
    <span class="example-tag" onclick="fillExample(this)">快递显示已签收，但家里没人，包裹也不见了</span>
  </div>
</div>

<script>
const MODELS = __MODELS__;
let current = null;

function renderModels() {
  const box = document.getElementById('modelBox');
  box.innerHTML = '';
  MODELS.forEach((m, i) => {
    const div = document.createElement('div');
    div.className = 'model-opt' + (m.available ? '' : ' disabled') + (m.default ? ' active' : '');
    if (m.default && m.available) current = m.key;
    const badge = m.device === 'GPU' ? '<span class="badge gpu">GPU</span>' : '<span class="badge">CPU</span>';
    div.innerHTML = '<div class="name">' + m.label + badge + '</div>'
      + '<div class="meta">F1 ' + m.macro_f1 + ' · ' + m.size_mb + ' MB'
      + (m.available ? '' : ' · ' + m.reason) + '</div>';
    if (m.available) {
      div.onclick = () => {
        current = m.key;
        document.querySelectorAll('.model-opt').forEach(e => e.classList.remove('active'));
        div.classList.add('active');
      };
    }
    box.appendChild(div);
  });
}

function fillExample(el) {
  document.getElementById('textInput').value = el.textContent;
  updateCount();
  doPredict();
}

function clearAll() {
  document.getElementById('textInput').value = '';
  document.getElementById('resultBox').classList.remove('show');
  updateCount();
}

function updateCount() {
  document.getElementById('charCount').textContent =
    document.getElementById('textInput').value.length;
}

async function doPredict() {
  const input = document.getElementById('textInput');
  const btn = document.getElementById('predictBtn');
  const box = document.getElementById('resultBox');
  const rows = document.getElementById('rowsBox');
  const text = input.value.trim();

  if (!current) { box.classList.add('show'); rows.innerHTML = '<span class="error">没有可用模型</span>'; return; }
  if (!text) { box.classList.add('show'); rows.innerHTML = '<span class="error">请输入投诉文本</span>'; return; }
  if (text.length > 500) { box.classList.add('show'); rows.innerHTML = '<span class="error">文本请控制在 500 字以内</span>'; return; }

  btn.disabled = true;
  btn.innerHTML = '<span class="loading"></span>';
  box.classList.remove('show');
  const t0 = performance.now();

  try {
    const resp = await fetch('/predict', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text, model: current })
    });
    const data = await resp.json();
    const rtt = Math.round(performance.now() - t0);
    if (data.error) {
      rows.innerHTML = '<span class="error">' + data.error + '</span>';
    } else {
      rows.innerHTML = data.top5.map((it, i) =>
        '<div class="row' + (i === 0 ? ' top1' : '') + '">'
        + '<div class="cat">' + it.label + '</div>'
        + '<div class="bar-wrap"><div class="bar" style="width:' + (it.prob * 100).toFixed(1) + '%"></div></div>'
        + '<div class="pct">' + (it.prob * 100).toFixed(2) + '%</div></div>').join('');
      document.getElementById('modelInfo').textContent =
        '模型：' + data.model_label + '（' + data.device + '）';
      document.getElementById('timeInfo').textContent =
        '服务端 ' + data.ms + ' ms · 往返 ' + rtt + ' ms';
    }
    box.classList.add('show');
  } catch (e) {
    rows.innerHTML = '<span class="error">请求失败：' + e.message + '</span>';
    box.classList.add('show');
  } finally {
    btn.disabled = false;
    btn.textContent = '预测';
  }
}

// 支持分享式演示链接：/?model=xxx&demo=文本&auto=1 打开即预填并自动预测
// 注意：必须在 renderModels() 之后调用，否则卡片还没渲染，高亮改不动
function applyDemoParams() {
  const q = new URLSearchParams(location.search);
  const demo = q.get('demo');
  const key = q.get('model');
  if (key) {
    const idx = MODELS.findIndex(x => x.key === key && x.available);
    if (idx >= 0) {
      current = MODELS[idx].key;
      document.querySelectorAll('.model-opt').forEach((e, i) =>
        e.classList.toggle('active', i === idx));
    }
  }
  if (demo) {
    document.getElementById('textInput').value = demo;
    updateCount();
    if (q.get('auto') === '1') doPredict();
  }
}

document.getElementById('textInput').addEventListener('input', updateCount);
document.getElementById('textInput').addEventListener('keydown', function (e) {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') doPredict();
});
renderModels();
applyDemoParams();
</script>
</body>
</html>
"""


def render(models):
    """把模型清单注入页面（models: [{key,label,device,macro_f1,size_mb,available,reason,default}]）。"""
    return PAGE.replace("__MODELS__", json.dumps(models, ensure_ascii=False))
