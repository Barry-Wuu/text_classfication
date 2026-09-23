# -*- coding: utf-8 -*-
"""导出可部署的量化权重并直传 GitHub Release（Kaggle -> GitHub，不经过本机）。

产物（上传到 Barry-Wuu/text_classfication 的 deploy-models release）：
  bert_nf4_state.pt     NF4 4bit 量化（encoder Linear -> bnb Linear4bit），约 110MB，需 GPU 推理
  bert_int8_state.pt    INT8 动态量化（Linear -> qint8），约 146MB，CPU 推理
  deploy_models_meta.json  加载说明与指标
"""
import sys, os, json, time, traceback

_LOG = open('/kaggle/working/run.log', 'w', encoding='utf-8')


class _Tee:
    def __init__(self, *fs): self.fs = fs

    def write(self, s):
        for f in self.fs:
            try: f.write(s); f.flush()
            except Exception: pass

    def flush(self):
        for f in self.fs:
            try: f.flush()
            except Exception: pass

    def isatty(self): return False

    def fileno(self):
        for f in self.fs:
            try: return f.fileno()
            except Exception: pass
        return -1

    @property
    def encoding(self): return 'utf-8'


sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)


def _exchook(t, v, tb):
    txt = ''.join(traceback.format_exception(t, v, tb))
    print('FATAL: ' + txt, flush=True)
    try: open('/kaggle/working/FATAL.txt', 'w', encoding='utf-8').write(txt)
    except Exception: pass


sys.excepthook = _exchook
os.environ.setdefault('TRANSFORMERS_NO_TF', '1')

import requests
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification

CKPT = "bert_epoch59.pt"
CKPT_URLS = [
    "https://github.com/Barry-Wuu/text_classfication/releases/download/bert-weights/" + CKPT,
    "https://github.com/Barry-Wuu/text_classfication/releases/download/bert-weights/" + CKPT,
]
CKPT_DST = "/tmp/" + CKPT
MODEL_PATH = "bert-base-chinese"
N_LABELS = 18

REPO = 'Barry-Wuu/text_classfication'
RELEASE_TAG = 'deploy-models'
RELEASE_NAME = '可部署模型权重（NF4 4bit / INT8 量化）'
TOKEN = os.environ.get('GH_TOKEN', '').strip()
if not TOKEN:
    raise SystemExit('缺少 GH_TOKEN：请在 Kaggle Add-ons → Secrets 注入，或在本地设置环境变量')
API = 'https://api.github.com'
UPLOAD = 'https://uploads.github.com'
HDR = {'Authorization': 'Bearer ' + TOKEN,
       'Accept': 'application/vnd.github+json',
       'X-GitHub-Api-Version': '2022-11-28'}


def download(urls, dst, min_size):
    if os.path.exists(dst) and os.path.getsize(dst) >= min_size:
        return dst
    for url in urls:
        try:
            pos = os.path.getsize(dst) if os.path.exists(dst) else 0
            h = {"Range": "bytes=%d-" % pos} if pos else {}
            r = requests.get(url, headers=h, stream=True, timeout=300)
            if r.status_code not in (200, 206):
                print("  HTTP", r.status_code, url[:60], flush=True); continue
            mode = "ab" if (pos and r.status_code == 206) else "wb"
            with open(dst, mode) as f:
                for c in r.iter_content(1 << 20):
                    f.write(c)
            if os.path.getsize(dst) >= min_size:
                print("  权重就绪", os.path.getsize(dst), flush=True); return dst
        except Exception as e:
            print("  下载异常", repr(e)[:100], flush=True)
    raise RuntimeError("权重下载失败")


def gh(path, method='GET', **kw):
    r = requests.request(method, API + path, headers=HDR, timeout=120, **kw)
    if r.status_code >= 400:
        print("  GH", method, path, r.status_code, r.text[:200], flush=True)
        return None
    return r.json() if r.text else {}


def ensure_release():
    rel = gh(f'/repos/{REPO}/releases/tags/{RELEASE_TAG}')
    if rel:
        print("  release 已存在:", rel['id'], flush=True); return rel
    rel = gh(f'/repos/{REPO}/releases', 'POST', json={
        'tag_name': RELEASE_TAG, 'name': RELEASE_NAME,
        'body': '投满分文本分类：可部署模型权重。NF4 4bit 需 GPU（bitsandbytes）；'
                'INT8 动态量化在 CPU 上跑。加载方式见 deploy_models_meta.json。'})
    print("  release 已创建:", rel['id'], flush=True)
    return rel


def upload(rel, path):
    name = os.path.basename(path)
    size = os.path.getsize(path)
    for a in rel.get('assets', []):
        if a['name'] == name:
            print(f"  跳过已存在: {name}", flush=True); return
    print(f"  上传 {name} ({size/1048576:.1f} MB)", flush=True)
    t0 = time.time()
    with open(path, 'rb') as f:
        r = requests.post(
            f"{UPLOAD}/repos/{REPO}/releases/{rel['id']}/assets?name={name}",
            headers={**HDR, 'Content-Type': 'application/octet-stream'}, data=f,
            timeout=1800)
    print(f"    -> {r.status_code} ({time.time()-t0:.0f}s)", flush=True)
    if r.status_code >= 400:
        print("    ", r.text[:200], flush=True)


def main():
    print("torch", torch.__version__, "| cuda", torch.cuda.is_available(), flush=True)
    ck = download(CKPT_URLS, CKPT_DST, 300 * 1024 * 1024)
    ckpt = torch.load(ck, map_location="cpu", weights_only=False)
    state = ckpt["state_dict"]

    # ---- INT8 动态量化（CPU 推理用） ----
    print("[1/2] INT8 动态量化", flush=True)
    m = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=N_LABELS)
    m.load_state_dict(state)
    m.eval()
    qi = torch.ao.quantization.quantize_dynamic(m.cpu(), {nn.Linear}, dtype=torch.qint8)
    p_int8 = '/kaggle/working/bert_int8_state.pt'
    torch.save({'state_dict': qi.state_dict(), 'num_labels': N_LABELS,
                'kind': 'int8_dynamic_linear', 'base': MODEL_PATH}, p_int8)
    print("  int8 体积 %.1f MB" % (os.path.getsize(p_int8) / 1048576), flush=True)

    # ---- NF4 4bit（GPU 推理用） ----
    p_nf4 = '/kaggle/working/bert_nf4_state.pt'
    if torch.cuda.is_available():
        try:
            import bitsandbytes as bnb
        except ImportError:
            import subprocess as sp
            sp.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=True)
            import bitsandbytes as bnb
        print("[2/2] NF4 4bit (bnb %s)" % bnb.__version__, flush=True)
        m4 = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=N_LABELS)
        for name, mod in list(m4.named_modules()):
            if isinstance(mod, nn.Linear) and ".encoder." in name + ".":
                parent = m4.get_submodule(name.rsplit(".", 1)[0])
                setattr(parent, name.rsplit(".", 1)[1],
                        bnb.nn.Linear4bit(mod.in_features, mod.out_features,
                                          bias=mod.bias is not None,
                                          compute_dtype=torch.float16,
                                          quant_type="nf4"))
        m4.load_state_dict(state, strict=False)
        m4.to("cuda").eval()
        torch.save({'state_dict': m4.state_dict(), 'num_labels': N_LABELS,
                    'kind': 'nf4_4bit_encoder_linear', 'base': MODEL_PATH,
                    'quant_type': 'nf4', 'compute_dtype': 'float16'}, p_nf4)
        print("  nf4 体积 %.1f MB" % (os.path.getsize(p_nf4) / 1048576), flush=True)
    else:
        print("[2/2] 跳过 NF4：无 GPU", flush=True)

    meta = {
        "repo": REPO, "release": RELEASE_TAG,
        "models": {
            "bert_nf4_state.pt": {
                "kind": "nf4_4bit_encoder_linear", "device": "cuda",
                "metrics": {"acc": 0.8570, "macro_f1": 0.8590, "size_mb": 108.1, "ms_per_sample_gpu": 2.09},
                "how_to_load": [
                    "import bitsandbytes as bnb, torch",
                    "from transformers import BertForSequenceClassification",
                    "m=BertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=18)",
                    "把 m 中 .encoder. 下的 nn.Linear 换成 bnb.nn.Linear4bit(...,quant_type='nf4',compute_dtype=torch.float16)",
                    "m.load_state_dict(torch.load('bert_nf4_state.pt',map_location='cpu')['state_dict'],strict=False)",
                    "m.to('cuda').eval()"],
            },
            "bert_int8_state.pt": {
                "kind": "int8_dynamic_linear", "device": "cpu",
                "metrics": {"acc": 0.8483, "macro_f1": 0.8518, "size_mb": 145.6, "ms_per_sample_cpu": 169.41},
                "how_to_load": [
                    "m=BertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=18)",
                    "m=torch.ao.quantization.quantize_dynamic(m,{torch.nn.Linear},dtype=torch.qint8)",
                    "m.load_state_dict(torch.load('bert_int8_state.pt',map_location='cpu')['state_dict'])"],
            },
        },
        "teacher": {"acc": 0.8628, "macro_f1": 0.8660, "size_mb": 390.3},
        "note": "NF4 需要 CUDA + bitsandbytes；INT8 只能在 CPU 上跑（PyTorch 动态量化的限制）。",
    }
    p_meta = '/kaggle/working/deploy_models_meta.json'
    json.dump(meta, open(p_meta, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    print("== 上传 GitHub Release ==", flush=True)
    rel = ensure_release()
    if rel:
        for p in (p_nf4, p_int8, p_meta):
            if os.path.exists(p):
                upload(rel, p)
    print("===RESULT_START===")
    print(json.dumps({"files": [(os.path.basename(p), round(os.path.getsize(p)/1048576, 1))
                                for p in (p_nf4, p_int8, p_meta) if os.path.exists(p)]},
                     ensure_ascii=False))
    print("===RESULT_END===")
    print("done.", flush=True)


if __name__ == "__main__":
    main()
