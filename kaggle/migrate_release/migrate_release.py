# -*- coding: utf-8 -*-
"""把组长仓库 Release 的权重资产迁移到个人仓库 Release。

背景:
  BERT 微调权重原先挂在组长仓库 Barry-Wuu/text_classfication 的
  bert-weights Release 上; 个人仓库 Barry-Wuu/text_classfication 是项目
  现在的正式归档, 需要把资产与 README 里的下载链接一并收敛到这里。

做法:
  Kaggle 云端直传(下载 -> 上传), 不经过本机 —— 4 个 .pt 合计约 1.56 GB,
  本机上行慢, 云端到 GitHub 是机房链路, 快得多。

产物:
  Barry-Wuu/text_classfication 的 bert-weights Release, 共 7 个资产。
"""
import sys
import os
import json
import time
import traceback

_LOG = open('/kaggle/working/run.log', 'w', encoding='utf-8')


class _Tee:
    def __init__(self, *fs):
        self.fs = fs

    def write(self, s):
        for f in self.fs:
            try:
                f.write(s)
                f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self.fs:
            try:
                f.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def fileno(self):
        for f in self.fs:
            try:
                return f.fileno()
            except Exception:
                pass
        return -1

    @property
    def encoding(self):
        return 'utf-8'


sys.stdout = _Tee(sys.__stdout__, _LOG)
sys.stderr = _Tee(sys.__stderr__, _LOG)


def _exchook(t, v, tb):
    txt = ''.join(traceback.format_exception(t, v, tb))
    print('FATAL: ' + txt, flush=True)
    try:
        open('/kaggle/working/FATAL.txt', 'w', encoding='utf-8').write(txt)
    except Exception:
        pass


sys.excepthook = _exchook

import requests

SRC_REPO = 'Barry-Wuu/text_classfication'
SRC_TAG = 'bert-weights'
SRC_BASE = f'https://github.com/{SRC_REPO}/releases/download/{SRC_TAG}/'

DST_REPO = 'Barry-Wuu/text_classfication'
DST_TAG = 'bert-weights'
DST_NAME = 'BERT 微调权重（text_classfication）'
DST_BODY = (
    '投满分文本分类：BERT 微调（打榜模型）与训练检查点。\n\n'
    '- `bert_epoch59.pt` — 打榜模型，前 100 轮里测试准确率最高'
    '（acc 0.8628 / macro-F1 0.8660）\n'
    '- `bert_epoch10/26/60.pt` — 滚动保留的检查点\n'
    '- `bert_best100.json` — 选模证据：前 100 轮逐轮指标\n'
    '- `result*.json` — 各阶段训练日志\n\n'
    '加载：`torch.load(...)` 取 `state_dict`，\n'
    '模型为 `BertForSequenceClassification`（bert-base-chinese，num_labels=18）。\n'
    '本 Release 原挂于组长仓库，现统一收敛到本仓库。'
)

# (文件名, 期望最小字节数) —— 期望值用于校验下载完整
ASSETS = [
    ('bert_epoch59.pt', 380 * 1048576),
    ('bert_epoch10.pt', 380 * 1048576),
    ('bert_epoch26.pt', 380 * 1048576),
    ('bert_epoch60.pt', 380 * 1048576),
    ('bert_best100.json', 8 * 1024),
    ('result.json', 200),
    ('result_stage2.json', 200),
]
TOKEN = os.environ.get('GH_TOKEN', '').strip()
if not TOKEN:
    raise SystemExit('缺少 GH_TOKEN：请在 Kaggle Add-ons → Secrets 注入，或在本地设置环境变量')
API = 'https://api.github.com'
UPLOAD = 'https://uploads.github.com'
HDR = {'Authorization': 'Bearer ' + TOKEN,
       'Accept': 'application/vnd.github+json',
       'X-GitHub-Api-Version': '2022-11-28'}

TMP = '/kaggle/working/dl'
os.makedirs(TMP, exist_ok=True)


def gh(path, method='GET', **kw):
    r = requests.request(method, API + path, headers=HDR, timeout=180, **kw)
    if r.status_code >= 400:
        print('  GH', method, path, r.status_code, r.text[:200], flush=True)
        return None
    return r.json() if r.text else {}


def ensure_release():
    rel = gh(f'/repos/{DST_REPO}/releases/tags/{DST_TAG}')
    if rel:
        print('  release 已存在 id=%s，资产 %d 个' % (rel['id'], len(rel.get('assets', []))),
              flush=True)
        return rel
    rel = gh(f'/repos/{DST_REPO}/releases', 'POST',
             json={'tag_name': DST_TAG, 'name': DST_NAME, 'body': DST_BODY})
    if not rel:
        raise RuntimeError('release 创建失败')
    print('  release 已创建 id=%s' % rel['id'], flush=True)
    return rel


def download(name, min_size, dst):
    """带续传的下载；返回实际大小。"""
    for attempt in range(4):
        have = os.path.getsize(dst) if os.path.exists(dst) else 0
        if have >= min_size:
            return have
        url = SRC_BASE + name
        h = {'Range': 'bytes=%d-' % have} if have else {}
        try:
            r = requests.get(url, headers=h, stream=True, timeout=600)
            if r.status_code not in (200, 206):
                print('    下载 HTTP', r.status_code, flush=True)
                time.sleep(5)
                continue
            mode = 'ab' if (have and r.status_code == 206) else 'wb'
            if mode == 'wb':
                have = 0
            got = have
            with open(dst, mode) as f:
                for c in r.iter_content(1 << 20):
                    f.write(c)
                    got += len(c)
                    if got % (100 << 20) < (1 << 20):
                        print('      %d/%d MB' % (got >> 20, min_size >> 20), flush=True)
            size = os.path.getsize(dst)
            if size >= min_size:
                return size
            print('    下载不完整 %d < %d' % (size, min_size), flush=True)
        except Exception as e:
            print('    下载异常', repr(e)[:110], flush=True)
            time.sleep(3 + 3 * attempt)
    raise RuntimeError('下载失败: ' + name)


def upload(rel, path, name):
    for a in rel.get('assets', []):
        if a['name'] == name:
            print('    已存在，跳过:', name, flush=True)
            return True
    size = os.path.getsize(path)
    t0 = time.time()
    with open(path, 'rb') as f:
        r = requests.post(
            '%s/repos/%s/releases/%s/assets?name=%s' % (UPLOAD, DST_REPO, rel['id'], name),
            headers={**HDR, 'Content-Type': 'application/octet-stream'},
            data=f, timeout=3600)
    print('    -> %d  %.0fs  %.1f MB' % (r.status_code, time.time() - t0, size / 1048576),
          flush=True)
    if r.status_code >= 400:
        print('    ', r.text[:250], flush=True)
        return False
    return True


def main():
    print('源: %s (%s)' % (SRC_REPO, SRC_TAG), flush=True)
    print('目标: %s (%s)' % (DST_REPO, DST_TAG), flush=True)
    rel = ensure_release()
    done, failed = [], []
    for name, min_size in ASSETS:
        print('[%s]' % name, flush=True)
        dst = os.path.join(TMP, name)
        try:
            size = download(name, min_size, dst)
            print('    下载完成 %.2f MB' % (size / 1048576), flush=True)
        except Exception as e:
            print('    失败:', repr(e)[:120], flush=True)
            failed.append((name, 'download'))
            continue
        if upload(rel, dst, name):
            done.append((name, round(size / 1048576, 2)))
        else:
            failed.append((name, 'upload'))
        try:
            os.remove(dst)          # 逐个释放磁盘，避免 4x390MB 堆积
        except Exception:
            pass

    print('===RESULT_START===')
    print(json.dumps({'uploaded': done, 'failed': failed}, ensure_ascii=False))
    print('===RESULT_END===')
    print('done.', flush=True)


if __name__ == '__main__':
    main()
