# -*- coding: utf-8 -*-
"""统一分词入口（供 joblib 序列化按引用记录）。

坑：直接把 `jieba.lcut` 交给 TfidfVectorizer 会导致模型无法保存——
`jieba.lcut` 是惰性加载的 Tokenizer 实例上的绑定方法，实例里带 RLock，
pickle 时报 TypeError: cannot pickle '_thread.RLock' object。
放进模块级函数后，pickle 只记录 `jieba_cut.cut` 这个引用，加载端导入同名模块即可。
"""
import jieba

jieba.setLogLevel(20)


def cut(text):
    return jieba.lcut(text)
