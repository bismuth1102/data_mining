import pandas as pd
import re
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
import tensorflow as tf

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score

def lowerCase(df):
    df.replace(regex={'<.*?>|&nbsp|\W': ' '}, inplace=True)
    df = df.fillna("miss")
    df['name'] = df['name'].str.lower()
    df['lvl1'] = df['lvl1'].str.lower()
    df['lvl2'] = df['lvl2'].str.lower()
    df['lvl3'] = df['lvl3'].str.lower()
    df['descrption'] = df['descrption'].str.lower()
    df['type'] = df['type'].str.lower()

    return df


def loadTest():
    df_test = pd.read_csv('data/test_data.csv')
    sLength = len(df_test['id'])
    
    df_test = lowerCase(df_test)

    df_test['score'] = pd.Series(np.zeros(sLength, dtype=np.int), index=df_test.index)
    return df_test


def loadTrain():
    df_train = pd.read_csv('data/train_data.csv')
    df_label = pd.read_csv('data/train_label.csv')
    df_train = df_train.set_index('id').join(df_label.set_index('id'), how='inner')
    
    df_train = lowerCase(df_train)
    
    return df_train


X_train =loadTrain()
X_test = loadTest()
y_score = X_train.score.values


def mySub(m):
    m = re.sub(r'&|:|<li>|</li>|ul>|</ul>|\+|\-|\/|\(|\)', '', m)
    m = re.sub(r'\s+\d\s+|\s+[a-zA-Z]\s+', '', m)
    return m


desKeys=[]
def desKey(df):
    dic={}
    corpus = np.array(df['name'])
    for item in corpus:
        item = mySub(item)
        words = item.split(" ")
        for word in words:
            if word in dic:
                dic[word] = dic[word]+1
            else:
                dic[word] = 1

    for each in dic:
        if dic[each]<200 and dic[each]>120:
            desKeys.append(each)


def desFilter(str):
    words=str.split(" ")
    result = ""
    for word in words:
        for key in desKeys:
            if word==key:
                result+=key
                result+=" "
    return result


def desText(df):
    desKey(df[['name']])
    df['name'] = df['name'].apply(desFilter)

    return df


def tfidf(df):
	# 先过滤得到中间频率词
	df['name'] = df.name.apply(mySub)
	df = desText(df.astype(str))

	# tfidf
	corpus = np.array(df['name'])
	vectorizer = CountVectorizer()
	X = vectorizer.fit_transform(corpus)
	words = vectorizer.get_feature_names()
	transformer = TfidfTransformer()
	tfidf = transformer.fit_transform(X)
	tfidf = tfidf.toarray()

	# 字典排序
	bag=[]
	for row in range(len(tfidf)):
		dict = {}
		for i in range(len(words)):
			dict[words[i]] = tfidf[row][i]
		dict_array = sorted(dict.items(), key=lambda e:e[1], reverse=True)
		bag.append(dict_array)

	# 找到每行tfidf最大的前3个
	tfidf_bag=[]
	for row in bag:
		tfidf=""
		i = 0
		for word in row:
			i = i+1
			tfidf+=word[0]
			if i > 2:
				break
		tfidf_bag.append(tfidf)

	df['name'] = tfidf_bag
	return df


X_train = tfidf(X_train)
X_test = tfidf(X_test)


df_train = pd.DataFrame(data=X_train['lvl1']+X_train['lvl2']+X_train['lvl3']+X_train['type']+X_train['name'],columns=['train'])
X_train_text = df_train.train.astype(str)

df_test = pd.DataFrame(data=X_test['lvl1']+X_test['lvl2']+X_test['lvl3']+X_test['type']+X_test['name'],columns=['test'])
X_test_text = df_test.test.astype(str)

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_train_seq = pad_sequences(X_train_seq,maxlen=8)

X_test_seq = tokenizer.texts_to_sequences(X_test_text)
X_test_seq = pad_sequences(X_test_seq, maxlen=8)


rfr = RandomForestRegressor(
    n_estimators= 70, 
    max_depth=40, 
    min_samples_split=55,
    min_samples_leaf=12,
    max_features=72,
    oob_score=True, 
    random_state=100)

rfr.fit(X_train_seq, y_score)
rfr_y_predict = rfr.predict(X_test_seq)

# 输出结果
Y2 = X_['id'].values
rfr_submission = pd.DataFrame({'id': Y2, 'Score': rfr_y_predict})
rfr_submission.to_csv('data/submission.csv', index=False)


