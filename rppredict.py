# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 16:31:09 2018

@author: Srikanth
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 14:17:11 2018

@author: Srikanth
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv('estimators_rp_raw.csv', sep=',',header=0)

df.loc[:,'Outs'] = np.sum([df.loc[:,'TBF'],df.loc[:,'H']*-1,df.loc[:,'BB']*-1,df.loc[:,'IBB']*-1,df.loc[:,'HBP']*-1],axis=0)
df.loc[:,'PPA'] = np.divide(np.sum([df.loc[:,'Outs'],df.loc[:,'SO'],df.loc[:,'H']*-1,df.loc[:,'BB']*-1,df.loc[:,'IBB']*-1,df.loc[:,'ER']*-2],axis=0),df.loc[:,'TBF'])
df.loc[:,'PPG'] = np.divide(np.sum([df.loc[:,'Outs'],df.loc[:,'W']*5,df.loc[:,'SV']*5,df.loc[:,'SO'],df.loc[:,'H']*-1,df.loc[:,'BB']*-1,df.loc[:,'IBB']*-1,df.loc[:,'ER']*-2],axis=0),df.loc[:,'G'])

plt.figure()
df['PPG'].plot.hist();

hr_pct = np.divide(df.loc[:,'HR'],df.loc[:,'TBF'])
endurance = np.divide(df.loc[:,'Outs'],df.loc[:,'G'])
efficiency = np.divide(df.loc[:,'Outs'],df.loc[:,'TBF'])

predictors = np.zeros([11,len(df)])

for i in range(0,len(df)):
    predictors[0,i] = float(df.loc[i,'K%'][0:4])/100
    predictors[1,i] = float(df.loc[i,'BB%'][0:4])/100
    predictors[2,i] = hr_pct[i]
    bip = 1-float(df.loc[i,'K%'][0:4])/100-float(df.loc[i,'BB%'][0:4])/100-np.divide(df.loc[i,'IBB'],df.loc[i,'TBF'])-hr_pct[i]
    predictors[3,i] = float(df.loc[i,'GB%'][0:4])*bip/100
    predictors[4,i] = float(df.loc[i,'LD%'][0:4])*bip/100
    predictors[5,i] = float(df.loc[i,'Soft%'][0:4])*bip/100
    predictors[6,i] = float(df.loc[i,'Hard%'][0:4])*bip/100
    predictors[7,i] = float(df.loc[i,'Pull%'][0:4])*bip/100
    predictors[8,i] = float(df.loc[i,'Oppo%'][0:4])*bip/100
    predictors[9,i] = endurance[i]
    predictors[10,i] = efficiency[i]
    
fip = np.zeros([3,len(df)])
fipficiency = np.zeros([5,len(df)])

for i in range(0,len(df)):
    fip[0,i] = float(df.loc[i,'K%'][0:4])/100
    fip[1,i] = float(df.loc[i,'BB%'][0:4])/100
    fip[2,i] = hr_pct[i]
    fipficiency[0,i] = float(df.loc[i,'K%'][0:4])/100
    fipficiency[1,i] = float(df.loc[i,'BB%'][0:4])/100
    fipficiency[2,i] = hr_pct[i]
    fipficiency[3,i] = endurance[i]
    fipficiency[4,i] = efficiency[i]
    
ppa = np.array(df.loc[:,'PPA'])
ppg = np.array(df.loc[:,'PPG'])

bbp_fit = np.linalg.lstsq(np.transpose(predictors),ppg,rcond=-1)
fip_fit = np.linalg.lstsq(np.transpose(fip),ppg,rcond=-1)
fipficiency_fit = np.linalg.lstsq(np.transpose(fipficiency),ppg,rcond=-1)

bbp_model = bbp_fit[0]
fip_model = fip_fit[0]
fipficiency_model = fipficiency_fit[0]

bbp_estimate = np.squeeze(np.asarray(np.matmul(bbp_model,predictors)))
fip_estimate = np.squeeze(np.asarray(np.matmul(fip_model,fip)))
fipficiency_estimate = np.squeeze(np.asarray(np.matmul(fipficiency_model,fipficiency)))

corr1 = np.corrcoef([bbp_estimate,ppa])[0,1]
corr2 = np.corrcoef([fip_estimate,ppa])[0,1]
corr3 = np.corrcoef([fipficiency_estimate,ppa])[0,1]

print(corr1*corr1,corr2*corr2,corr3*corr3)

df = pd.read_csv('estimators_rp_test.csv', sep=',',header=0)

df.loc[:,'Outs'] = np.sum([df.loc[:,'TBF'],df.loc[:,'H']*-1,df.loc[:,'BB']*-1,df.loc[:,'IBB']*-1,df.loc[:,'HBP']*-1],axis=0)
df.loc[:,'PPA'] = np.divide(np.sum([df.loc[:,'Outs'],df.loc[:,'SO'],df.loc[:,'H']*-1,df.loc[:,'BB']*-1,df.loc[:,'IBB']*-1,df.loc[:,'ER']*-2],axis=0),df.loc[:,'TBF'])
df.loc[:,'PPG'] = np.divide(np.sum([df.loc[:,'Outs'],df.loc[:,'W']*5,df.loc[:,'SV']*5,df.loc[:,'SO'],df.loc[:,'H']*-1,df.loc[:,'BB']*-1,df.loc[:,'IBB']*-1,df.loc[:,'ER']*-2],axis=0),df.loc[:,'G'])

plt.figure()
df['PPG'].plot.hist();

hr_pct = np.divide(df.loc[:,'HR'],df.loc[:,'TBF'])
endurance = np.divide(df.loc[:,'Outs'],df.loc[:,'G'])
efficiency = np.divide(df.loc[:,'Outs'],df.loc[:,'TBF'])

predictors = np.zeros([11,len(df)])

for i in range(0,len(df)):
    predictors[0,i] = float(df.loc[i,'K%'][0:4])/100
    predictors[1,i] = float(df.loc[i,'BB%'][0:4])/100
    predictors[2,i] = hr_pct[i]
    bip = 1-float(df.loc[i,'K%'][0:4])/100-float(df.loc[i,'BB%'][0:4])/100-np.divide(df.loc[i,'IBB'],df.loc[i,'TBF'])-hr_pct[i]
    predictors[3,i] = float(df.loc[i,'GB%'][0:4])*bip/100
    predictors[4,i] = float(df.loc[i,'LD%'][0:4])*bip/100
    predictors[5,i] = float(df.loc[i,'Soft%'][0:4])*bip/100
    predictors[6,i] = float(df.loc[i,'Hard%'][0:4])*bip/100
    predictors[7,i] = float(df.loc[i,'Pull%'][0:4])*bip/100
    predictors[8,i] = float(df.loc[i,'Oppo%'][0:4])*bip/100
    predictors[9,i] = endurance[i]
    predictors[10,i] = efficiency[i]
    
fip = np.zeros([3,len(df)])
fipficiency = np.zeros([5,len(df)])

for i in range(0,len(df)):
    fip[0,i] = float(df.loc[i,'K%'][0:4])/100
    fip[1,i] = float(df.loc[i,'BB%'][0:4])/100
    fip[2,i] = hr_pct[i]
    fipficiency[0,i] = float(df.loc[i,'K%'][0:4])/100
    fipficiency[1,i] = float(df.loc[i,'BB%'][0:4])/100
    fipficiency[2,i] = hr_pct[i]
    fipficiency[3,i] = endurance[i]
    fipficiency[4,i] = efficiency[i]
    
ppa = np.array(df.loc[:,'PPA'])
ppg = np.array(df.loc[:,'PPG'])

bbp_estimate = np.squeeze(np.asarray(np.matmul(bbp_model,predictors)))
fip_estimate = np.squeeze(np.asarray(np.matmul(fip_model,fip)))
fipficiency_estimate = np.squeeze(np.asarray(np.matmul(fipficiency_model,fipficiency)))

corr1 = np.corrcoef([bbp_estimate,ppa])[0,1]
corr2 = np.corrcoef([fip_estimate,ppa])[0,1]
corr3 = np.corrcoef([fipficiency_estimate,ppa])[0,1]

print(corr1*corr1,corr2*corr2,corr3*corr3)