# ### 导入数据
import numpy as np
import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import time
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

data = pd.read_excel('.\data.xlsx')
data.head()

def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

data['cut_comment'] = data.comment.apply(chinese_word_cut)
data.head()

# #### 提取特征
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
                       token_pattern=u'(?u)\\b[^\\dd\\W]\\w+\\b',
                       stop_words=frozenset(stopwords),
                       sublinear_tf=True)
# vect = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.8, max_features=100000)

# #### 划分数据集
#划分数据集
X = data['cut_comment']
y = data.sentiment

ACC = []
Recall = []
Fscore = []
Precesion = []

for num in range(1,21):
    randomState = num
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomState)

    # flag = 80
    # for index, value in y_train.items():
    #     if value == 1 and flag > 0:
    #         flag = flag - 1
    #         y_train[index] = 0

    #特征展示
    test = pd.DataFrame(vect.fit_transform(X_train).toarray(), columns=vect.get_feature_names())
    test.head()

    # #### 训练模型
    nb = MultinomialNB()

    X_train_vect = vect.fit_transform(X_train)
    nb.fit(X_train_vect, y_train)
    train_score = nb.score(X_train_vect, y_train)

    # #### 测试模型
    X_test_vect = vect.transform(X_test)

    y_pred = nb.predict(X_test_vect)
    # y_pre_proba = nb.predict_proba(X_test_vect)

    ACC.append(metrics.accuracy_score(y_test, y_pred))
    Recall.append(metrics.recall_score(y_test, y_pred))
    Fscore.append(metrics.f1_score(y_test, y_pred))
    Precesion.append(metrics.precision_score(y_test, y_pred))

print("ACC")
print(ACC)
print("%.4f" % (sum(ACC)/len(ACC)))
print("Recall")
print(Recall)
print("%.4f" % (sum(Recall) / len(Recall)))
print("Fscore")
print(Fscore)
print("%.4f" % (sum(Fscore) / len(Fscore)))
print("Precesion")
print(Precesion)
print("%.4f" % (sum(Precesion) / len(Precesion)))