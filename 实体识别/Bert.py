import random
import jsonlines

fp = []
train_data = []
dev_data = []
test_data = []
with jsonlines.open('D:\MyProject\学术研究\Research\learn_pytorch\实体识别\\admin.jsonl', 'r') as fp1:
    for p in fp1:
        fp.append(p)
# train_data, dev_data = numpy.split(data, 200)

flag = 0
# random.shuffle(fp)
for p in fp:
    p['text'] = p['text'].replace(' ', ',')
    if flag < 3:
        train_data.append(p)
    elif flag < 200:
        dev_data.append(p)
    else:
        test_data.append(p)
    flag = flag + 1

import re

label_type = {'o': 0, '部门': 1, '污染物': 2, '产品': 3, '标准法规': 4}


def decode_label(d):
    text_len = len(d['text'])
    label = [0] * text_len
    for t in d['label']:
        type = t[2]
        st = t[0]
        ed = t[1]
        for j in range(st, ed):
            label[j] = label_type[type]
    return label


def transfrom_data(data, mode):
    data_texts1 = [re.sub('\d', '&', d['text']) for d in data]
    data_texts = [re.sub('[a-zA-Z]', '&', d) for d in data_texts1]

    if mode == 'train':
        data_labels = []
        for d in data:
            data_labels.append(decode_label(d))
        return (data_texts, data_labels)

    else:
        return data_texts


train_texts, train_labels = transfrom_data(train_data, 'train')
dev_texts, dev_labels = transfrom_data(dev_data, 'train')
test_texts, test_labels = transfrom_data(train_data, 'train')

from transformers import BertTokenizer
from IPython.display import clear_output

# 使用bert的tokenizer将文字转化成数字。
PRETRAINED_MODEL_NAME = "bert-base-chinese"  # 指定为中文
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path='bert-base-chinese')
clear_output()

train_ids = []
dev_ids = []
test_ids = []
for train_text in train_texts:
    train_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(train_text)))

for dev_text in dev_texts:
    dev_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dev_text)))

for test_text in test_texts:
    test_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(test_text)))

print(train_ids[0])
print(dev_texts[0])
print(dev_labels[0])

import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

MaxLen = 100


class NewDataset(Dataset):
    def __init__(self, ids, labels):
        self.ids = ids
        self.labels = labels
        self.len = len(ids)

    def __getitem__(self, item):
        tokens_tensor = torch.tensor(self.ids[item])
        label_tensor = torch.tensor(self.labels[item])
        return (tokens_tensor, label_tensor)

    def __len__(self):
        return self.len


trainset = NewDataset(train_ids, train_labels)
devset = NewDataset(dev_ids, dev_labels)
BATCH_SIZE = 1


def create_mini_batch(samples):
    tokens_tensors = [s[0] for s in samples]
    label_tensors = [s[1] for s in samples]

    # zero pad 到同一序列长度
    one = [0]
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    label_tensors = pad_sequence(label_tensors, batch_first=True, padding_value=0)

    if len(tokens_tensors[0]) != MaxLen:
        tokens_tensors = torch.tensor([t + one for t in tokens_tensors.numpy().tolist()])
    if len(label_tensors[0]) != MaxLen:
        label_tensors = torch.tensor([t + one for t in label_tensors.numpy().tolist()])
    # attention masks，将 tokens_tensors 不为 zero padding 的位置设为1
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)

    return tokens_tensors, masks_tensors, label_tensors


trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch, drop_last=False)
devloader = DataLoader(devset, batch_size=BATCH_SIZE, collate_fn=create_mini_batch, drop_last=False)

from transformers import BertForTokenClassification

model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=17)
device = torch.device("cpu")
model.to(device)

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
Epochs = 10
for epoch in range(Epochs):
    losses = 0.0
    for data in trainloader:
        tokens_tensors, masks_tensors, label_tensors = [t.to(device) for t in data]
        optimizer.zero_grad()
        outputs = model(input_ids=tokens_tensors, attention_mask=masks_tensors, labels=label_tensors)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        losses += loss.item()
    print(losses)

import numpy as np


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


nb_eval_steps = 0
model.eval()
eval_loss, eval_accuracy = 0, 0
predictions, true_labels = [], []

for data in devloader:
    tokens_tensors, masks_tensors, label_tensors = [t.to(device) for t in data]
    with torch.no_grad():
        outputs = model(input_ids=tokens_tensors, attention_mask=masks_tensors, labels=label_tensors)
        loss = outputs[0]
        preds = model(input_ids=tokens_tensors, attention_mask=masks_tensors)

    for pred, label_tensor in zip(preds[0], label_tensors):
        logit = pred.detach().cpu().numpy()  # detach的方法，将variable参数从网络中隔离开，不参与参数更新
        label_ids = label_tensor.cpu().numpy()

        predictions.extend(np.argmax(logit, axis=1))
        true_labels.extend(label_ids)
        # 计算accuracy 和 loss
        tmp_eval_accuracy = flat_accuracy(logit, label_ids)

        eval_loss += loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1

print("Validation loss: {}".format(eval_loss / nb_eval_steps))
print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

from sklearn.metrics import f1_score

pred_tags = list(np.array(predictions).flatten())
valid_tags = list(np.array(true_labels).flatten())
print(pred_tags[0:20])
print(valid_tags[0:20])
print("F1-Score: {}".format(f1_score(pred_tags, valid_tags, average='weighted')))

test_texts, test_labels = transfrom_data(test_data, 'train')
test_ids = []
test_true = []
test_pred = []
for test_text in test_texts:
    test_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(test_text)))
flag = 0
for p in test_ids:
    test_true.extend(test_labels[flag])
    print(test_data[flag]['text'])
    print(test_labels[flag])
    test_tokens_tensor = torch.tensor(p)
    test_tokens_tensor = test_tokens_tensor

    test_masks_tensor = torch.zeros(test_tokens_tensor.shape, dtype=torch.long)
    test_masks_tensor = test_masks_tensor.masked_fill(test_tokens_tensor != 0, 1)

    outputs = model(input_ids=test_tokens_tensor.unsqueeze(0).cpu(),
                    attention_mask=test_masks_tensor.unsqueeze(0).cpu())
    logits = outputs[0]
    preds = []
    for logit in logits:
        preds.extend(np.argmax(logit.detach().cpu().numpy(), axis=1))

    inverse_dict = dict([val, key] for key, val in label_type.items())
    #     preds1 = [inverse_dict[i] for i in preds]

    test_pred.extend(preds)
    print(preds)
    flag = flag + 1

from sklearn import metrics

label = [0, 1, 2, 3, 4]

for i in range(len(test_true),-1,-1 ):
    if test_pred[i] == 0 and test_true[i] == 0:
        del test_pred[i]
        del test_true[i]

print('ACC: %.4f' % metrics.accuracy_score(test_true, test_pred))
print('Recall: %.4f' % metrics.recall_score(test_true, test_pred, labels=label, average="macro"))
print('F1-score: %.4f' % metrics.f1_score(test_true, test_pred, labels=label, average="macro"))
print('Precesion: %.4f' % metrics.precision_score(test_true, test_pred, labels=label, average="macro"))
