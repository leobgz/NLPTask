#!/usr/bin/env python
# coding: utf-8

# 来源：https://blog.csdn.net/weixin_42617035/article/details/102680583  <br>
# https://blog.csdn.net/google19890102/article/details/25592833 （贝叶斯）假设特征相互独立 <br>
# https://blog.csdn.net/google19890102/article/details/80091502 （snownlp原理讲解） <br>
# https://blog.csdn.net/weixin_41961559/article/details/105237852 (推荐了工具)<br>

# ### 导入数据
import numpy as np
import pandas as pd
data = pd.read_excel('.\data.xlsx')
data.head()

# ### 朴素贝叶斯
# #### 数据预处理
#根据需要做处理
#去重、去除停用词

# #### jieba分词
import jieba

def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

data['cut_comment'] = data.comment.apply(chinese_word_cut)
data.head()

# #### 提取特征
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
def get_custom_stopwords(stop_words_file):
    with open(stop_words_file) as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list

stop_words_file = '哈工大停用词表.txt'
stopwords = get_custom_stopwords(stop_words_file)

# vect = TfidfVectorizer(max_df = 0.8,
#                        min_df = 3,
#                        token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
#                        stop_words=frozenset(stopwords),
#                        sublinear_tf=True)
vect = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=100000)

# #### 划分数据集
#划分数据集
X = data['cut_comment']
y = data.sentiment

from sklearn.model_selection import train_test_split
import time
randomState = time.localtime(time.time()).tm_sec
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

#特征展示
test = pd.DataFrame(vect.fit_transform(X_train).toarray(), columns=vect.get_feature_names())
test.head()

# #### 训练模型

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

X_train_vect = vect.fit_transform(X_train)
nb.fit(X_train_vect, y_train)
train_score = nb.score(X_train_vect, y_train)
print(train_score)

# #### 测试模型
X_test_vect = vect.transform(X_test)
print(nb.score(X_test_vect, y_test))

y_pred = nb.predict(X_test_vect)
y_pre_proba = nb.predict_proba(X_test_vect)

from sklearn import metrics
print(randomState)
print('AUC: %.4f' % metrics.roc_auc_score(y_test, y_pred))
print('ACC: %.4f' % metrics.accuracy_score(y_test, y_pred))
print('Recall: %.4f' % metrics.recall_score(y_test, y_pred))
print('F1-score: %.4f' % metrics.f1_score(y_test, y_pred))
print('Precesion: %.4f' % metrics.precision_score(y_test, y_pred))

# # #### 分析数据
# data = pd.read_excel(".\sample_data2.xlsx")
# data.head()
#
#
# data = pd.read_excel(".\sample_data2.xlsx")
# data['cut_comment'] = data.comment.apply(chinese_word_cut)
# X=data['cut_comment']
# X_vec = vect.transform(X)
# nb_result = nb.predict(X_vec)
# #predict_proba(X)[source] 返回概率
# data['nb_result'] = nb_result
#
#
# test = pd.DataFrame(vect.fit_transform(X).toarray(), columns=vect.get_feature_names())
#
# # data.to_excel("data_result_idf.xlsx",index=False)
