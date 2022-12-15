import fasttext
import pandas as pd
from sklearn.model_selection import train_test_split
import jieba
import time

# 数据处理

data = pd.read_excel('sample_data.xlsx')


def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))


data['cut_comment'] = data.comment.apply(chinese_word_cut)
data['sentiment'] = data['sentiment'].map(lambda x: str(int(x)))
data['fastText'] = data['cut_comment'].str.cat(data['sentiment'], sep='\t__label__')

# data.to_excel("data_result.xlsx",index=False)

ft = data['fastText']

train_path = 'data_train.txt'
test_path = 'data_test.txt'
randomState = time.localtime(time.time()).tm_sec
train, test = train_test_split(ft, test_size=0.2, random_state=randomState)
data_train = pd.DataFrame(train)
data_train.to_csv('data_train.txt', sep='\n', index=0, header=0)
data_test = pd.DataFrame(test)
data_test.to_csv('data_test.txt', sep='\n', index=0, header=0)

model = fasttext.train_supervised(train_path)
print(model.test(test_path))
y_pred = model.predict(train_path)
print(y_pred)