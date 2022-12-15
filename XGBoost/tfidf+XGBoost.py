# ### 导入数据
import numpy as np
import pandas as pd
import time
from sklearn import metrics
import jieba
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

data = pd.read_excel('data.xlsx')
data.head()


# 数据预处理
# 去重、去除停用词
# jieba分词
def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))


data['cut_comment'] = data.comment.apply(chinese_word_cut)
data.head()


# 提取特征
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
vect = CountVectorizer(max_df=0.8,
                       min_df=3,
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                       stop_words=frozenset(stopwords))

# 划分数据集
X = data['cut_comment']
y = data.sentiment


randomState = time.localtime(time.time()).tm_sec
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

# 特征展示
test = pd.DataFrame(vect.fit_transform(X_train).toarray(), columns=vect.get_feature_names())
test.head()

# XGBoost
import xgboost as xgb

X_train_vect = vect.fit_transform(X_train)
X_test_vect = vect.transform(X_test)
dtrain = xgb.DMatrix(X_train_vect, label=y_train)
# dval =xgb.DMatrix(X_test_vect, label=y_test)
dtest = xgb.DMatrix(X_test_vect)
param = {'eval_metric': 'logloss'}
# watchlist =[(dtrain,'train'),(dval,'val')]
watchlist = [(dtrain, 'train')]
# num_rounds=500
model = xgb.train(param, dtrain, evals=watchlist, early_stopping_rounds=10)  # 训练
# make prediction
ypred = model.predict(dtest)
# 设置阈值, 输出一些评价指标
y_pred = (ypred >= 0.5) * 1

print('AUC: %.4f' % metrics.roc_auc_score(y_test, ypred))
print('ACC: %.4f' % metrics.accuracy_score(y_test, y_pred))
print('Recall: %.4f' % metrics.recall_score(y_test, y_pred))
print('F1-score: %.4f' % metrics.f1_score(y_test, y_pred))
print('Precesion: %.4f' % metrics.precision_score(y_test, y_pred))
metrics.confusion_matrix(y_test, y_pred)
