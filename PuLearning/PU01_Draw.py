from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
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

data_train = pd.read_excel('data_train.xlsx')
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