import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import time
import numpy as np
from pulearn import ElkanotoPuClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer

def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

data_train = pd.read_excel('data_train2.xlsx')
data_test = pd.read_excel('data_test.xlsx')
data_train['cut_comment'] = data_train.comment.apply(chinese_word_cut)
data_test['cut_comment'] = data_test.comment.apply(chinese_word_cut)

X_train = data_train['cut_comment']
y_train = data_train.sentiment
X_test = data_test['cut_comment']
y_test = data_test.sentiment

def get_custom_stopwords(stop_words_file):
    with open(stop_words_file) as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list
stop_words_file = '哈工大停用词表.txt'
stopwords = get_custom_stopwords(stop_words_file)

vect = TfidfVectorizer(max_df = 0.8,
                       min_df = 3,
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                       stop_words=frozenset(stopwords))
X_train_vect = vect.fit_transform(X_train)
X_test_vect = vect.transform(X_test)

svc = SVC(C=10, kernel='rbf', gamma=0.4, probability=True)
pu_estimator = ElkanotoPuClassifier(estimator=svc)
pu_estimator.fit(X_train_vect, y_train)

data_test['result'] = pu_estimator.predict(X_test_vect)

y_pred = data_test['result']
y_pre_proba = pu_estimator.predict_proba(X_test_vect)

print('AUC: %.4f' % metrics.roc_auc_score(y_test, y_pred))
print('ACC: %.4f' % metrics.accuracy_score(y_test, y_pred))
print('Recall: %.4f' % metrics.recall_score(y_test, y_pred))
print('F1-score: %.4f' % metrics.f1_score(y_test, y_pred))
print('Precesion: %.4f' % metrics.precision_score(y_test, y_pred))

data_test.to_excel("data_result.xlsx", index=False)