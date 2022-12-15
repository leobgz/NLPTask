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

data = pd.read_excel('sample_data.xlsx')
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

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, labels, title='Confusion Matrix'):
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=90)
    plt.yticks(xlocations, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


label = ["Low level", "High level"]
tick_marks = np.array(range(len(label))) + 0.5
y_true = y_test

cm = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(label))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.0:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=17, va='center', ha='center')

# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm_normalized, label, title='Normalized confusion matrix')
# plt.savefig('../Data/confusion_matrix.png', format='png')
plt.show()