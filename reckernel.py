import json
import csv
import pandas as pd
from collections import defaultdict
from collections import Counter
import numpy as np



def getOriginRecItems(rec,style):

    recs=set()
    for i in rec:
        for j in i["items"]:
            recs.add(j[:-2])
    needRec=set(style.keys())-recs
    return needRec,  list(recs)
    
class NearestNeighbors(object):
    def __init__(self,n_neighbors):
        self.n=n_neighbors
        self.db=np.array([],dtype=int)
    def fit(self,Y):
        self.db=Y
        return self
    def kneighbors(self,X):
        query=np.tile(X, (len(self.db),1))
        dist=np.zeros(len(self.db),dtype=int)
        x,y=np.where(query==self.db)
        cpt=zip(x,y)
        for i in cpt:
            dist[i[0]]+=X[i[1]]
        index=np.argsort(dist)
        dist=dist[index][:-(self.n)-1:-1]
        index=index[:-(self.n)-1:-1]
        return dist,index
def nearest(recs,distances,indices,sub,i):
    if len(distances)==0 or distances[0]<=1:
        return

    print(distances,indices)

    for j in np.where(distances==distances[0])[0]:#mode2
        sub[i].append(recs[indices[j]])
        # if distances[0]<=2:
            # print(i,style[i]['name'],recs[indices[j]],style[recs[indices[j]]]['name'])

def itemColorSearch(filename):
    with open(filename) as f:
        prd=json.load(f)
    prd_color=defaultdict(dict)
    for i in prd:
        prd_color[i['styleNo']][i['id']]=i["color"]
    return prd_color

def colorRules_subLineNo(data,rec):
    color_rules=defaultdict(list)
    rec_search=defaultdict(list)
    for i in range(len(rec)):
        unodr_class=[str(data.loc[int(k[:-2])]["class"]) for k in rec[i]["items"]]
        order=np.argsort(unodr_class)
        rec[i]["items"]=np.array(rec[i]["items"])[order]

        rec[i]["class"]=''.join(np.array(unodr_class)[order])
        colors=[]
        for j in rec[i]["items"]:
            rec_search[str(j)[:-2]].append(i)
            colors.append(prd_color[j[:-2]][j])
        color_rules[rec[i]["class"]].append(",".join(colors))

    for i in color_rules:
        fre=Counter(color_rules[i]).most_common()
        if fre[-1][1]==1:
            fre.pop()
        color_rules[i]=list(set(color_rules[i]))
        for j in range(len(color_rules[i])):
            color_rules[i][j]=color_rules[i][j].split(",")
    return color_rules, rec_search

def ruleSub(color_rules,rec,row,mode,newRule,i=0,j=0):
    rules=color_rules[rec[row]["class"]]
    to_sub=np.copy(rec[row]["items"])
    tosub_colors=[]
    for k in range(len(to_sub)):
        if mode==0 and to_sub[k][:5]==str(j):
            to_sub[k]=i
        else:
            to_sub[k]=to_sub[k][:5]
        tosub_colors.append(list(prd_color[to_sub[k]].keys()))
        if len(list(prd_color[to_sub[k]].keys()))==0:
            print(to_sub[k])

    for r in rules:
        sub_prd=[]
        k,p=0,0
        while k<len(r):
            compare_color=prd_color[tosub_colors[k][p][:-2]][tosub_colors[k][p]]
            if compare_color==r[k]:
                sub_prd.append([tosub_colors[k][p],p])
                k+=1
                if k==len(r):
                    break
                p=0
            else:
                p+=1
            while p==len(tosub_colors[k]):
                if k!=len(r)-1 and len(sub_prd)!=0:
                    p=sub_prd.pop()[1]+1
                    k-=1
                else:
                    k=len(r)
                    break
        if len(sub_prd)==len(r):

            if mode==1 and [i[0] for i in sub_prd]==list(rec[row]["items"]):
                pass
            # newRule[i]={"category":'alg',"items":[i[0] for i in sub_prd],'ref':rec[q]['items']}
            newRule.append({"category":'alg',"items":[i[0] for i in sub_prd],'ref':list(rec[row]['items'])})



if __name__ == '__main__':
    data = pd.read_csv('class.v3.csv', delimiter=',',index_col=0)
    with open('./crawler/style.v2.json') as f:
        style=json.load(f)
    with open('./crawler/recommend.v2.json') as f:
        rec=json.load(f)

    # 產品顏色查找
    prd_color=itemColorSearch('./crawler/products.v2.json')
    # 顏色規則、已有推薦的商品所在原推薦行數
    colorRules,subLineNo=colorRules_subLineNo(data,rec)

    newRule=[]
    """Part.1 把原有推薦的品項組合查找其還可能的不同顏色組合"""
    for line in range(len(rec)):
        ruleSub(colorRules,rec,line,1,newRule)

    """Part.2 knn找相似的商品，再過濾顏色規則"""
    # 需要生推薦的品項、已有的推薦品項
    needRec, recItems=getOriginRecItems(rec,style)

    # 把原有推薦的分類(男女/部位)
    recsItemsClass = [[] for i in range(18)]
    for i in recItems :
        if style[i]['target']=='women':
            recsItemsClass[data.loc[int(i)]['class']].append(i)
        elif style[i]['target']=='men':
            recsItemsClass[data.loc[int(i)]['class']+9].append(i)
        else:
            recsItemsClass[data.loc[int(i)]['class']+9].append(i)
            recsItemsClass[data.loc[int(i)]['class']].append(i)

    # 分類執行knn
    nbrs = [NearestNeighbors(n_neighbors=4).fit(data.loc[[int(j) for j in recsItemsClass[i]]].as_matrix()[:,:-1]) for i in range(18)]
    sub=defaultdict(list)
    for i in needRec:
        _class=data.loc[int(i)]['class']#0~8
        if _class==0:
            continue
        X = data.loc[int(i)].as_matrix()[:-1]
        if style[i]["target"]=="women":
            distances, indices = nbrs[_class].kneighbors(X)    
            # pass
        elif style[i]["target"]=="men":
            _class+=9
            distances, indices = nbrs[_class].kneighbors(X)    
        else:
            fdistances, findices = nbrs[_class].kneighbors(X)    
            mdistances, mindices = nbrs[_class+9].kneighbors(X) 
            if fdistances[0]>mdistances[0]:
                distances,indices=fdistances,findices
            elif mdistances[0]>fdistances[0]:
                distances,indices=mdistances,mindices
            else:
                if recsItemsClass[_class][findices[0]]!=recsItemsClass[_class+9][mindices[0]]:
                    nearest(recsItemsClass[_class],fdistances,findices,sub,i)
                    distances,indices=mdistances,mindices
                    _class+=9
        nearest(recsItemsClass[_class],distances,indices,sub,i)

    noSimStyle=set(needRec)-set(sub.keys())#沒有相似的品項

    # 顏色規則過濾
    for i in sub.items():
        for j in i[1]:
            for line in subLineNo[str(j)]:
                ruleSub(colorRules,rec,line,0,newRule,i[0],j)


    for i in newRule:
        print([prd_color[j[:-2]][j] for j in i['items']])
        print([style[j[:-2]]['name'] for j in i['items']])
        print([j for j in i['items']])
        print([style[j[:-2]]['name'] for j in i['ref']])
        print([j for j in i['ref']])
        print("-----------------------------")
    with open('newRcm.v2.json', 'w')as f:
        json.dump(newRule, f, sort_keys=True, indent=4, ensure_ascii=False)
    print(len(newRule))
