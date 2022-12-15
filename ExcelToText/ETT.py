import pandas as pd
import numpy as np

data = pd.read_excel('.\data.xlsx')
i = 1

def save(filename, contents):
 fh = open(filename, 'w', encoding='utf-8')
 fh.write(contents)
 fh.close()

for p in data["comment"]:
    st = str(i) + ".txt"
    p = p.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace(u'&nbsp;', ' ').replace(u'\xa0', '').replace('　', ' ')
    save(st, p)
    i = i + 1
