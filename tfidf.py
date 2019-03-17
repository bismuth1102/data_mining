# coding:utf-8
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import re
from pandas import DataFrame

def mySub(m):
    m = re.sub(r'&|:|<li>|</li>|ul>|</ul>', '', m)
    # m = re.sub(r'\s+\d\s+|\s+[a-zA-Z]\s+|\s+(are|font-size|It|by|at|up|color|will|No|not|any|that|"<ul|<td|0px|padding|<p)\s+', ' ', m)
   
    return m

nameKeys=[]
def nameKey(df):
    dic={}
    corpus = np.array(df['descrption'])
    for item in corpus:
        item = mySub(item)
        words = item.split(" ")
        for word in words:
            if word in dic:
                dic[word] = dic[word]+1
            else:
                dic[word] = 1

    for each in dic:
        if dic[each]>0:
            nameKeys.append(each)
    print(nameKeys)


def nameFilter(str):
    words=str.split(" ")
    result = ""
    for word in words:
        for key in nameKeys:
            if word==key:
                result+=key
                result+=" "
    return result


def process(df):
    nameKey(df[['descrption']])
    df[(True-df['descrption'].isin(nameKeys))]
    # df['descrption'] = df['descrption'].apply(nameFilter)
    return df

 
# 先过滤得到中间频率词
df = pd.read_csv('data/train.csv')
df['descrption'] = df.descrption.apply(mySub)
df = process(df.astype(str))

corpus = np.array(df['descrption'])
# print (corpus)

# 将文本中的词语转换为词频矩阵
vectorizer = CountVectorizer()
#计算个词语出现的次数
X = vectorizer.fit_transform(corpus)
#获取词袋中所有文本关键词
words = vectorizer.get_feature_names()
#查看词频结果
# print(X.toarray())
 
from sklearn.feature_extraction.text import TfidfTransformer
 
#类调用
transformer = TfidfTransformer()
# print(transformer)
#将词频矩阵X统计成TF-IDF值
tfidf = transformer.fit_transform(X)
#查看数据结构 tfidf[i][j]表示i类文本中的tf-idf权重

tfidf = tfidf.toarray()

bag=[]
for row in range(len(tfidf)):
	dict = {}
	for i in range(len(words)):
		# print(words[i])
		# print(tfidf[row][i])
		dict[words[i]] = tfidf[row][i]
	dict_array = sorted(dict.items(), key=lambda e:e[1], reverse=True)
	bag.append(dict_array)

tfidf_bag=[]
for row in bag:
	tfidf=[]
	i = 0
	for word in row:
		i = i+1
		tfidf.append(word[0])
		if i > 3:
			break
	tfidf_bag.append(tfidf)

ser = pd.Series(tfidf_bag)
df['descrption'] = ser

print(df['descrption'])

# for i in range(len(tfidf_bag)):
# 	nameKeys = tfidf_bag[i]
# 	print(nameKeys)
# 	result = nameFilter(df['descrption'][i])
# 	print(result)

# print(DataFrame(tfidf.toarray()))

