import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_excel('.\data.xlsx')
data.head()

# ### 朴素贝叶斯
# #### 数据预处理
# 根据需要做处理
# 去重、去除停用词

# #### jieba分词
import jieba


def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))


data['cut_comment'] = data.comment.apply(chinese_word_cut)
data.head()

# #### 提取特征
from sklearn.feature_extraction.text import TfidfVectorizer


def get_custom_stopwords(stop_words_file):
    with open(stop_words_file) as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list


stop_words_file = '哈工大停用词表.txt'
stopwords = get_custom_stopwords(stop_words_file)

vect = TfidfVectorizer(max_df=0.8,
                       min_df=3,
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                       stop_words=frozenset(stopwords),
                       sublinear_tf=True)

# #### 划分数据集
# 划分数据集
X = data['cut_comment']
y = data.sentiment

from sklearn.model_selection import train_test_split, cross_val_score
import time

randomState = time.localtime(time.time()).tm_sec
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=100000)
vectorizer = TfidfVectorizer(max_df = 0.8,
                       min_df = 3,
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                       stop_words=frozenset(stopwords),
                       sublinear_tf=True)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

clf = DecisionTreeClassifier(random_state=randomState)
rfc = RandomForestClassifier(random_state=randomState)

clf = clf.fit(X_train_vect, y_train)
rfc = rfc.fit(X_train_vect, y_train)
score_c = clf.score(X_test_vect, y_test)
score_r = rfc.score(X_test_vect, y_test)

y_pred = clf.predict(X_test_vect)
y_pred2 = rfc.predict(X_test_vect)

from sklearn import metrics
print("决策树")
print('AUC: %.4f' % metrics.roc_auc_score(y_test, y_pred))
print('ACC: %.4f' % metrics.accuracy_score(y_test, y_pred))
print('Recall: %.4f' % metrics.recall_score(y_test, y_pred))
print('F1-score: %.4f' % metrics.f1_score(y_test, y_pred))
print('Precesion: %.4f' % metrics.precision_score(y_test, y_pred))

print("随机森林")
print('AUC: %.4f' % metrics.roc_auc_score(y_test, y_pred2))
print('ACC: %.4f' % metrics.accuracy_score(y_test, y_pred2))
print('Recall: %.4f' % metrics.recall_score(y_test, y_pred2))
print('F1-score: %.4f' % metrics.f1_score(y_test, y_pred2))
print('Precesion: %.4f' % metrics.precision_score(y_test, y_pred2))
