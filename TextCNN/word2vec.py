import jieba
import re
import numpy as np
from sklearn.decomposition import PCA
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

data = pd.read_excel('sample_data.xlsx')
lines = []
for line in data.comment:  # 分别对每段分词
    temp = jieba.lcut(line)  # 结巴分词 精确模式
    words = []
    for i in temp:
        # 过滤掉所有的标点符号
        i = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+", "", i)
        if len(i) > 0:
            words.append(i)
    if len(words) > 0:
        lines.append(words)
print(lines[0:5])  # 预览前5行分词结果

# 调用Word2Vec训练
# 参数：size: 词向量维度；window: 上下文的宽度，min_count为考虑计算的单词的最低词频阈值
# model = Word2Vec(lines, vector_size=20, window=2, min_count=1, epochs=7, negative=10, sg=1)
model = Word2Vec(lines, vector_size=2, window=2, min_count=1)
print("食品的词向量：\n", model.wv.get_vector('食品'))
print("\n和食品相关性最高的前20个词语：")
print(model.wv.most_similar('食品', topn=20))  # 与孔明最相关的前20个词语