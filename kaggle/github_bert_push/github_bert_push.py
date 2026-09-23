# -*- coding: utf-8 -*-
"""把上游 BERT 内核的权重推到 GitHub Release（全程 Kaggle -> GitHub，不经过本机）。

为什么走 Release 而不是 commit：GitHub 普通仓库单文件硬上限 100MB，而 epoch 权重约 400MB；
Release asset 单文件上限 2GB，是搬运大权重的正确通道。

链路：wubarry/textcls-bert-finetune 的 output（kernel_sources 挂载）
      -> 本脚本 -> GitHub Releases API（uploads.github.com 流式上传）

只搬 TARGETS 里的文件，其余只打印清单。
"""
import hashlib
import json
import os
import sys
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

REPO = 'Barry-Wuu/text_classfication'
RELEASE_TAG = 'bert-weights'
RELEASE_NAME = 'BERT 微调权重（text_classfication）'
UPSTREAM_SLUG = 'textcls-bert-continue2'
TARGETS = ['bert_epoch59.pt', 'result.json']
TOKEN = os.environ.get('GH_TOKEN', '').strip()
if not TOKEN:
    raise SystemExit('缺少 GH_TOKEN：请在 Kaggle Add-ons → Secrets 注入，或在本地设置环境变量')
API = 'https://api.github.com'
UPLOAD = 'https://uploads.github.com'
HDR = {
    'Authorization': 'Bearer ' + TOKEN,
    'Accept': 'application/vnd.github+json',
    'X-GitHub-Api-Version': '2022-11-28',
    'User-Agent': 'kaggle-relay',
}


def md5(p, cap=None):
    h = hashlib.md5()
    n = 0
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
            n += len(b)
            if cap and n >= cap:
                break
    return h.hexdigest()


def find_mount():
    roots = ['/kaggle/input/notebooks/wubarry/' + UPSTREAM_SLUG,
             '/kaggle/input/' + UPSTREAM_SLUG]
    for r in roots:
        if os.path.isdir(r):
            print('  mount(root):', r, flush=True)
            found = {}
            for cur, _, fs in os.walk(r):
                for f in fs:
                    found.setdefault(f, os.path.join(cur, f))
            return found
    found = {}
    for cur, _, fs in os.walk('/kaggle/input'):
        for f in fs:
            found.setdefault(f, os.path.join(cur, f))
    return found


def ensure_release():
    """返回 release id；不存在则创建。"""
    r = requests.get(f'{API}/repos/{REPO}/releases/tags/{RELEASE_TAG}',
                     headers=HDR, timeout=60)
    if r.status_code == 200:
        j = r.json()
        print('  复用已有 release:', j['id'], j.get('html_url'), flush=True)
        return j['id']
    if r.status_code != 404:
        print('  查询 release 返回', r.status_code, r.text[:200], flush=True)
    body = {
        'tag_name': RELEASE_TAG,
        'name': RELEASE_NAME,
        'body': '由 Kaggle 内核 textcls-bert-continue2/continue3 产出的 bert-base-chinese 微调权重。'
                '包含每 epoch 的 state_dict（.pt）与 result.json（逐轮指标）。',
        'draft': False,
        'prerelease': False,
    }
    r = requests.post(f'{API}/repos/{REPO}/releases', headers=HDR,
                      data=json.dumps(body), timeout=60)
    print('  创建 release ->', r.status_code, flush=True)
    if r.status_code not in (200, 201):
        raise RuntimeError('创建 release 失败: %s %s' % (r.status_code, r.text[:300]))
    j = r.json()
    print('  release id:', j['id'], j.get('html_url'), flush=True)
    return j['id']


def list_assets(rel_id):
    r = requests.get(f'{API}/repos/{REPO}/releases/{rel_id}/assets',
                     headers=HDR, params={'per_page': 100}, timeout=60)
    r.raise_for_status()
    return {a['name']: a for a in r.json()}


def del_asset(asset_id):
    r = requests.delete(f'{API}/repos/{REPO}/releases/assets/{asset_id}',
                        headers=HDR, timeout=60)
    return r.status_code


def upload(rel_id, path, name, size):
    url = f'{UPLOAD}/repos/{REPO}/releases/{rel_id}/assets'
    hdr = dict(HDR)
    hdr['Content-Type'] = 'application/octet-stream'
    hdr['Content-Length'] = str(size)
    t0 = time.time()
    with open(path, 'rb') as fh:
        r = requests.post(url, headers=hdr, params={'name': name},
                          data=fh, timeout=(60, 7200))
    dt = time.time() - t0
    return r, dt


def main():
    print('=' * 68, flush=True)
    print('repo:', REPO, '| tag:', RELEASE_TAG, flush=True)
    if not TOKEN:
        raise RuntimeError('GH_TOKEN 未设置')
    print('token len:', len(TOKEN), flush=True)

    me = requests.get(f'{API}/user', headers=HDR, timeout=60)
    print('auth ->', me.status_code, me.json().get('login') if me.status_code == 200 else me.text[:150],
          flush=True)

    found = find_mount()
    print('\n挂载点文件清单（共 %d 个）:' % len(found), flush=True)
    for k in sorted(found):
        print('   %-26s %12d B' % (k, os.path.getsize(found[k])), flush=True)

    rel_id = ensure_release()
    assets = list_assets(rel_id)
    print('  现有 assets:', {k: v['size'] for k, v in assets.items()}, flush=True)

    result = {}
    for tgt in TARGETS:
        print('\n' + '=' * 68, flush=True)
        print('JOB:', tgt, flush=True)
        if tgt not in found:
            print('  挂载点未找到该文件', flush=True)
            result[tgt] = {'ok': False, 'stage': 'locate'}
            continue
        src = found[tgt]
        size = os.path.getsize(src)
        if assets.get(tgt, {}).get('size') == size:
            print('  已在 Release，尺寸一致，跳过  size=%d' % size, flush=True)
            result[tgt] = {'ok': True, 'skipped': True, 'size': size,
                           'url': assets[tgt]['browser_download_url']}
            continue
        if tgt in assets:
            print('  同名 asset 尺寸不一致，先删除旧的 id=%s' % assets[tgt]['id'], flush=True)
            print('  delete ->', del_asset(assets[tgt]['id']), flush=True)
        m = md5(src)
        print('  上传 %s  size=%d md5=%s' % (tgt, size, m), flush=True)
        try:
            r, dt = upload(rel_id, src, tgt, size)
        except Exception as e:
            print('  上传异常:', str(e)[:250], flush=True)
            result[tgt] = {'ok': False, 'stage': 'upload', 'err': str(e)[:250]}
            continue
        print('  HTTP %s  用时 %.1fs  速率 %.0f KB/s' % (r.status_code, dt, size / 1024 / max(dt, .01)),
              flush=True)
        if r.status_code not in (200, 201):
            print('  失败响应:', r.text[:300], flush=True)
            result[tgt] = {'ok': False, 'stage': 'upload', 'code': r.status_code,
                           'err': r.text[:200]}
            continue
        j = r.json()
        print('  已上传:', j['name'], j['size'], 'B', flush=True)
        print('  URL:', j['browser_download_url'], flush=True)
        result[tgt] = {'ok': True, 'size': size, 'md5': m, 'upload_s': round(dt, 1),
                       'url': j['browser_download_url']}

    print('\n最终 assets:', {k: v['size'] for k, v in list_assets(rel_id).items()}, flush=True)
    print('===RESULT_START===')
    print(json.dumps(result, ensure_ascii=False))
    print('===RESULT_END===')
    print('done.', flush=True)


if __name__ == '__main__':
    main()
