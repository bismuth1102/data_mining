# 找出频繁词

import re
import pandas as pd
import numpy as np

# f=open("data/train_data.csv",'r')

def mySub(m):
    m = re.sub(r'&|:|<li>|</li>|ul>|</ul>|\+|\-|\/|\(|\)', '', m)
    m = re.sub(r'\s+\d\s+|\s+[a-zA-Z]\s+', '', m)
    # m = re.sub(r'\s+(For|for|With|with|and|to|in|of|Set|amp;|ol|size|the|hp|u003e|li|that|p|non|will|an|not|at|by|pu|up|out|it|are|as|from|this|can|you|be|no|on)\s+', ' ', m)
    return m

# ll=f.read()
# '''将空格都取代为逗号，方便后面的split（）'''
# ll = myReplace(ll)
# ll=ll.replace(" ",',') 
# '''防止由于文档编辑不规范出现双逗号的情况'''
# ll=ll.replace(",,",',')
# l=ll.split("\n")
# rows=[]
# dic={}
# for i in l:
#     row=i.split(",")
#     rows.append(row)
# for ii in rows:
#     for each in ii:
#         if each in dic:
#             dic[each]=dic[each]+1
#         else:
#             dic[each]=1


dic={}
train = pd.read_csv('data/train_data.csv')
corpus = np.array(train['name'].astype(str))
for item in corpus:
    item = mySub(item)
    words = item.split(" ")
    for word in words:
        if word in dic:
            dic[word] = dic[word]+1
        else:
            dic[word] = 1



#输出所有的排序：
dic = sorted(dic.items(),key=lambda x:x[1],reverse=True)
print(dic)
# keys=[]
# for each in dic:
#     if dic[each]<2500 and dic[each]>1800:
#         keys.append(each)
# print(keys)


# '''只输出最大的值'''
# HighValue=0
# HighKey=None
# for each in dic:
#     if dic[each]>HighValue:
#         HighValue=dic[each]
#         HighKey=each
# print(HighKey,HighValue)
