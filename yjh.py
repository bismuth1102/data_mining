import pandas as pd
import re
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
import tensorflow as tf

def load():
    df_test = pd.read_csv('data/test_data.csv')
    sLength = len(df_test['id'])
    df_test.replace(regex={'<.*?>|&nbsp|\W': ' '}, inplace=True)
    df_test = df_test.fillna("miss")
    df_test['name'] = df_test['name'].str.lower()
    df_test['lvl1'] = df_test['lvl1'].str.lower()
    df_test['lvl2'] = df_test['lvl2'].str.lower()
    df_test['lvl3'] = df_test['lvl3'].str.lower()
    df_test['descrption'] = df_test['descrption'].str.lower()
    df_test['type'] = df_test['type'].str.lower()
    df_test['score'] = pd.Series(np.zeros(sLength, dtype=np.int), index=df_test.index)
    return df_test

def loadY():
    df = pd.read_csv(file_name)
    return df

def loadX():
    df_train = pd.read_csv('data/train_data.csv')
    df_label = pd.read_csv('data/train_label.csv')
    df_train = df_train.set_index('id').join(df_label.set_index('id'), how='inner')
    df_train.replace(regex={'<.*?>|&nbsp|\W': ' '}, inplace=True)
    df_train['name'] = df_train['name'].str.lower()
    df_train['lvl1'] = df_train['lvl1'].str.lower()
    df_train['lvl2'] = df_train['lvl2'].str.lower()
    df_train['lvl3'] = df_train['lvl3'].str.lower()
    df_train['descrption'] = df_train['descrption'].str.lower()
    df_train['type'] = df_train['type'].str.lower()
    
    # df['score'] = df_label['score']
    df_train = df_train.fillna("miss")
    
    # df.dropna(inplace=True)
    # df['score'] = pd.Series(df_label['score'], index=df.index)
    return df_train


X_train =loadX()
X_test = load()

df_train = pd.DataFrame(data=X_train['lvl1']+X_train['lvl2']+X_train['lvl3']+X_train['type']+X_train['name']+X_train['descrption'],columns=['train'])
X_train_text = df_train.train

df_test = pd.DataFrame(data=X_test['name']+X_test['lvl1']+X_test['lvl2']+X_test['lvl3']+X_test['type']+X_test['descrption'],columns=['test'])
X_test_text = df_test.test

tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(X_train_text)

X_train_hh = tokenizer.texts_to_sequences(X_train_text)
X_train_hh = pad_sequences(X_train_hh,maxlen=256)

X_test_hh = tokenizer.texts_to_sequences(X_test_text)
X_test_hh = pad_sequences(X_test_hh, maxlen=256)

y_score = X_train.score.values

model = keras.Sequential()
model.add(keras.layers.Embedding(20000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(32, activation=tf.nn.relu))
model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])


model.fit(X_train_hh[2001:],y_score[2001:],
          epochs=11,
          batch_size=128,
          validation_data=(X_train_hh[0:2000], X_train[0:2000].score.values),
          verbose=1)

output_array = model.predict(X_test_hh)
print(output_array)

df_end_id = pd.DataFrame(data=X_test.id)
# print(X_test.id)

df_end_score = pd.DataFrame(data=output_array,columns=['score'])
df_frames = [df_end_id,df_end_score]
df_end = pd.concat(df_frames,axis=1)

df_end.to_csv('data/submission.csv',index=False)
# print(df_end)
# print(df_end.shape)


