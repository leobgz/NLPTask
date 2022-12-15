import pandas as pd

def spaceReplace(i):
    i = i.replace('  ', ' ')
    i = spaceReplace(i) if '  ' in i else i
    return i

data = pd.read_excel('data(文本).xlsx')
List = [""]
for p in data["comment"]:
    p = p.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace(u'&nbsp;', ' ').replace(u'\xa0', '').replace('　',' ')
    p = spaceReplace(p)
    List.append(p)
List.remove("")
data["文本"] = List
data.to_excel("data(文本)_chuli.xlsx", index=False)

