from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
# 定义函数
def TF_IDF(corpus):
    vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
    x = vectorizer.fit_transform(corpus)
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
    word_location = vectorizer.vocabulary_  # 词的位置
    weight=tfidf.toarray()#tf-idf权重矩阵
    return weight,word_location,x.toarray()

# 调用函数
# 这里做分词，使用空格隔开
corpus = [
            '我 来到 北京 清华大学',
            '他 来到 了 中国',
            '小明 硕士 毕业 与 中国 科学院',
            '我 爱 北京 天安门'
           ]
weight,word_location,tf = TF_IDF(corpus)
print(weight)
print(word_location)
print(tf)
