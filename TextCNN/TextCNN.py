import jieba
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import time

sequence_length = 0
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
    len_w = len(words)
    if len_w > 0:
        lines.append(words)
        if len_w > sequence_length:
            sequence_length = len_w
X = lines
y = data.sentiment
for line in X:
    if len(line) < sequence_length:
        for i in range(sequence_length-len(line)):
            line.append("pad")

randomState = time.localtime(time.time()).tm_sec
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=randomState)


dtype = torch.FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
sentences = X_train
labels = y_train  # 1 is good, 0 is not good.

# TextCNN Parameter
embedding_size = 2
num_classes = 2  # 0 or 1
batch_size = len(sentences)


ls = []
for wd in sentences:
    ls.append(" ".join(wd))
# word_list = " ".join(sentences).split()
word_list = " ".join(ls)
vocab = list(set(word_list.split()))
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)


def make_data(st, labels):
    inputs = []
    for sen in st:
        inputs.append([word2idx[n] for n in sen])

    targets = []
    for out in labels:
        targets.append(out)  # To using Torch Softmax Loss function
    return inputs, targets


input_batch, target_batch = make_data(sentences, labels)
input_batch, target_batch = torch.LongTensor(input_batch), torch.LongTensor(target_batch)

dataset = Data.TensorDataset(input_batch, target_batch)
loader = Data.DataLoader(dataset, batch_size, True)


class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN, self).__init__()
        self.W = nn.Embedding(vocab_size, embedding_size)
        output_channel = 3
        self.conv = nn.Sequential(
            # conv : [input_channel(=1), output_channel, (filter_height, filter_width), stride=1]
            nn.Conv2d(1, output_channel, (2, embedding_size)),
            nn.ReLU(),
            # pool : ((filter_height, filter_width))
            nn.MaxPool2d((2, 1)),
        )
        # fc
        self.fc = nn.Linear(output_channel, num_classes)

    def forward(self, X):
        '''
        X: [batch_size, sequence_length]
        '''
        batch_size = X.shape[0]
        embedding_X = self.W(X)  # [batch_size, sequence_length, embedding_size]
        embedding_X = embedding_X.unsqueeze(1)  # add channel(=1) [batch, channel(=1), sequence_length, embedding_size]
        conved = self.conv(embedding_X)  # [batch_size, output_channel, 1, 1]
        flatten = conved.view(batch_size, -1)  # [batch_size, output_channel*1*1]
        output = self.fc(flatten)
        return output


model = TextCNN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(5000):
    for batch_x, batch_y in loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        if (epoch + 1) % 1000 == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Test

ypred = []
for tt in X_test:
    test_text = tt
    tests = [[word2idx[n] for n in test_text]]
    test_batch = torch.LongTensor(tests).to(device)
    # Predict
    model = model.eval()
    predict = model(test_batch).data.max(1, keepdim=True)[1]
    ypred.append(predict[0][0])


from sklearn import metrics
print('AUC: %.4f' % metrics.roc_auc_score(y_test, ypred))
print('ACC: %.4f' % metrics.accuracy_score(y_test, ypred))
print('Recall: %.4f' % metrics.recall_score(y_test, ypred))
print('F1-score: %.4f' % metrics.f1_score(y_test, ypred))
print('Precesion: %.4f' % metrics.precision_score(y_test, ypred))
metrics.confusion_matrix(y_test, ypred)
