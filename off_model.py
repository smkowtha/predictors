# -*- coding: utf-8 -*-
import numpy as np, pandas as pd
"""
Goals of this program: 
    1 insert player profile
    2 return range of outcomes based on BABIP
"""

df=pd.read_csv('off_15-17.csv', sep=',',header=0)

year = np.array(df.iloc[:,0])
playerName = df.iloc[:,1]
games = df.iloc[:,3]
pa = df.iloc[:,4]
k = df.iloc[:,5]
bb = df.iloc[:,6]
single = df.iloc[:,7]
double = df.iloc[:,8]
triple = df.iloc[:,9]
hr = df.iloc[:,10]
ibb = df.iloc[:,11]
sb = df.iloc[:,12]
cs = df.iloc[:,13]
runs = df.iloc[:,14]
rbi = df.iloc[:,15]

        
totalbases = np.sum([single,double*2,triple*3,hr*4],axis=0)
discipline = np.sum([bb,k*-0.5,ibb],axis=0)
speed = np.sum([sb*2,cs*-1],axis=0)
points = np.sum([totalbases,discipline,speed,runs,rbi],axis=0)
ppa = np.divide(points,pa)

pct_stats = df.iloc[:,17:22]
k_pct = np.zeros(1318)
bb_pct = np.zeros(1318)
gb_pct = np.zeros(1318)
ld_pct = np.zeros(1318)
fb_pct = np.zeros(1318)

for i in range(0,1318):
    k_pct[i] = float(pct_stats.iloc[i,0][0:4])
    bb_pct[i] = float(pct_stats.iloc[i,1][0:4])
    gb_pct[i] = float(pct_stats.iloc[i,2][0:4])
    ld_pct[i] = float(pct_stats.iloc[i,3][0:4])
    fb_pct[i] = float(pct_stats.iloc[i,4][0:4])

sp_pct = (sb-cs)*100/(single+double+triple+hr+bb+ibb)
predictors = np.stack([k_pct,bb_pct,ld_pct,fb_pct,sp_pct])

ops = df.loc[:,'OPS']
ops_predict = np.stack([ops,sp_pct])
ops_fit = np.linalg.lstsq(np.transpose(ops_predict),ppa,rcond=-1)
bbp_fit = np.linalg.lstsq(np.transpose(predictors),ppa,rcond=-1)

ops_model = np.array(ops_fit[0])[np.newaxis]
bbp_model = np.array(bbp_fit[0])[np.newaxis]

ops_estimate = np.squeeze(np.asarray(np.matmul(ops_model,ops_predict)))
bbp_estimate = np.squeeze(np.asarray(np.matmul(bbp_model,predictors)))

print(np.corrcoef([ops_estimate,bbp_estimate,ppa]))
