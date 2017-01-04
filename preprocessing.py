import numpy as np
import json
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict

# 把拆好的詞做成0/1 table , 如data/attributeTable.v3.csv 

with open('./data/styleWithJieba.json') as f:
    style=json.load(f)
attr,styleNo=[],[]
for each in style:
    styleNo.append(each)
    attr.append(' '.join(style[each]['namejieba']))

vectorizer = CountVectorizer(stop_words=[],token_pattern="\S+")
counts = vectorizer.fit_transform(attr)
counts=counts.toarray()
styleNo=np.array(styleNo).transpose()

terms=vectorizer.get_feature_names()
terms.insert(0,'id')

attr=[]   
for each in style:
    tmp=style[each]
    attr.append(' '.join([tmp['category'],tmp['subcat'],tmp['pattern']]))
vectorizer = CountVectorizer(stop_words=[],token_pattern="\S+")
counts2 = vectorizer.fit_transform(attr)
counts2=counts.toarray()
terms2=vectorizer.get_feature_names()

data=np.column_stack((styleNo,counts,counts2))

with open('./data/attributeTable.csv', 'w', newline='') as fp:
    fp.write(','.join(terms)+','+','.join(terms2)+'\n')
    writer = csv.writer(fp, delimiter=',')
    writer.writerows(data)







data = pd.read_csv('test.csv', delimiter=',',index_col=0)
data2 = pd.read_csv('cat2.csv', delimiter=',',index_col=0)
# 之後合併兩張表
"""But it's still FAR from being ready to run algorithm. There're similar or unclear attributes in the table and which would decrease the accuracy. And further more it's needed to add a column 'class' to indicate the class label for every products. In summary, it's necessary to have a well-established attribute table as the file "data/attributeTable.v3.csv" before rununing the recommendation algorithm. And that definately requires a great artificial effort."""

