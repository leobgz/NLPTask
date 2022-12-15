from docx import Document
import glob
import os
import csv

def get_paragraphs(docx_path):
    # 打开word文档
    document = Document(docx_path)

    # 获取所有段落
    all_paragraphs = document.paragraphs
    paragraph_texts = []
    # 循环读取列表
    for paragraph in all_paragraphs:
        paragraph_texts.append(paragraph.text)

    return paragraph_texts

def spaceReplace(i):
    i = i.replace('  ', ' ')
    i = spaceReplace(i) if '  ' in i else i
    return i

files = glob.glob(os.path.join("*.docx"))
log_path = 'temp.csv'
file = open(log_path, 'a+', encoding='utf-8_sig', newline='')
csv_writer = csv.writer(file)
csv_writer.writerow([f'标题', '文本'])

print(files)
for filename in files:
    docx_path = filename
    str = ""
    paragraph_texts = get_paragraphs(docx_path)
    for item in paragraph_texts:
        str += ' ' + item
    str = str.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').replace(u'&nbsp;', ' ')
    str = spaceReplace(str)

    # 将文本中 '\n' 去除，然后写入txt文件中查看
    # with open(docx_path[:-4] + 'txt', 'w', encoding='utf-8') as file:
    #     file.write(str)
    #     for item in paragraph_texts:
    #         if not item:
    #             continue
    #         # print(item)
    #         file.write(item.replace("\n", "") + '\n')
    row = [docx_path[:-5],str]
    csv_writer.writerow(row)

file.close()
