# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 10:03:03 2018

@author: Srikanth

Program that reads an estimators file and returns a points value prediction
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv('estimators_build_raw.csv', sep=',',header=0)

tb = np.sum([df.loc[:,'1B'],df.loc[:,'2B']*2,df.loc[:,'3B']*3,df.loc[:,'HR']*4],axis=0)
discipline = np.sum([df.loc[:,'SO']*-0.5,df.loc[:,'BB'],df.loc[:,'IBB']],axis=0)
speed = np.sum([df.loc[:,'SB']*2,df.loc[:,'CS']*-1],axis=0)
ppa = np.divide(np.sum([tb,discipline,speed],axis=0),df.loc[:,'PA'])
ppg = np.divide(np.sum([tb,discipline,speed],axis=0),df.loc[:,'G'])

net_sb = np.sum([df.loc[:,'SB'],df.loc[:,'CS']*-1],axis=0);
obp =np.sum([df.loc[:,'1B'],df.loc[:,'2B'],df.loc[:,'3B'],df.loc[:,'HR'],df.loc[:,'BB'],df.loc[:,'IBB']],axis=0);
spd = np.divide(net_sb,obp)

hr_pct = np.divide(df.loc[:,'HR'],df.loc[:,'PA'])

predictors = np.zeros([10,len(df)])

for i in range(0,len(df)):
    predictors[0,i] = float(df.loc[i,'K%'][0:4])/100
    predictors[1,i] = float(df.loc[i,'BB%'][0:4])/100
    predictors[2,i] = hr_pct[i]
    bip = 1-float(df.loc[i,'K%'][0:4])/100-float(df.loc[i,'BB%'][0:4])/100-np.divide(df.loc[i,'IBB'],df.loc[i,'PA'])-hr_pct[i]
    predictors[3,i] = float(df.loc[i,'LD%'][0:4])*bip/100
    predictors[4,i] = float(df.loc[i,'FB%'][0:4])*bip/100
    predictors[5,i] = float(df.loc[i,'Soft%'][0:4])*bip/100
    predictors[6,i] = float(df.loc[i,'Hard%'][0:4])*bip/100
    predictors[7,i] = float(df.loc[i,'Pull%'][0:4])*bip/100
    predictors[8,i] = float(df.loc[i,'Oppo%'][0:4])*bip/100
    predictors[9,i] = spd[i]
    

bbp_fit = np.linalg.lstsq(np.transpose(predictors),ppa,rcond=-1)
bbp_fit_g = np.linalg.lstsq(np.transpose(predictors),ppg,rcond=-1)

bbp_model = np.array(bbp_fit[0])[np.newaxis]

bbp_estimate = np.squeeze(np.asarray(np.matmul(bbp_model,predictors)))

corr = np.corrcoef([bbp_estimate,ppa])[0,1]
r_sq = corr*corr

print('K% weight:',bbp_model[0,0],'BB% weight:',bbp_model[0,1])
print('LD% weight:',bbp_model[0,3],'FB% weight:',bbp_model[0,4])
print('Soft% weight:',bbp_model[0,5],'Hard% weight:',bbp_model[0,6])
print('Pull% weight:',bbp_model[0,7],'Oppo% weight:',bbp_model[0,8])
print('Speed% weight:',bbp_model[0,9],'HR% weight:',bbp_model[0,2])


df_test = pd.read_csv('estimators_build_test.csv', sep =',',header=0)

test_tb = np.sum([df_test.loc[:,'1B'],df_test.loc[:,'2B']*2,df_test.loc[:,'3B']*3,df_test.loc[:,'HR']*4],axis=0)
test_discipline = np.sum([df_test.loc[:,'SO']*-0.5,df_test.loc[:,'BB'],df_test.loc[:,'IBB']],axis=0)
test_speed = np.sum([df_test.loc[:,'SB']*2,df_test.loc[:,'CS']*-1],axis=0)
test_ppa = np.divide(np.sum([test_tb,test_discipline,test_speed],axis=0),df_test.loc[:,'PA'])

test_predict = np.zeros([10,len(df_test)])

test_net_sb = np.sum([df_test.loc[:,'SB'],df_test.loc[:,'CS']*-1],axis=0);
test_obp =np.sum([df_test.loc[:,'1B'],df_test.loc[:,'2B'],df_test.loc[:,'3B'],df_test.loc[:,'HR'],df_test.loc[:,'BB'],df_test.loc[:,'IBB']],axis=0);
test_spd = np.divide(test_net_sb,test_obp)
test_hr_pct = np.divide(df_test.loc[:,'HR'],df_test.loc[:,'PA'])

for i in range(0,len(df_test)):
    test_predict[0,i] = float(df_test.loc[i,'K%'][0:4])/100
    test_predict[1,i] = float(df_test.loc[i,'BB%'][0:4])/100
    test_predict[2,i] = test_hr_pct[i]
    bip = 1-float(df_test.loc[i,'K%'][0:4])/100-float(df_test.loc[i,'BB%'][0:4])/100-np.divide(df_test.loc[i,'IBB'],df_test.loc[i,'PA'])-test_hr_pct[i]
    test_predict[3,i] = float(df_test.loc[i,'LD%'][0:4])*bip/100
    test_predict[4,i] = float(df_test.loc[i,'FB%'][0:4])*bip/100
    test_predict[5,i] = float(df_test.loc[i,'Soft%'][0:4])*bip/100
    test_predict[6,i] = float(df_test.loc[i,'Hard%'][0:4])*bip/100
    test_predict[7,i] = float(df_test.loc[i,'Pull%'][0:4])*bip/100
    test_predict[8,i] = float(df_test.loc[i,'Oppo%'][0:4])*bip/100
    test_predict[9,i] = test_spd[i]

bbp_test = np.squeeze(np.asarray(np.matmul(bbp_model,test_predict)))
corr_test = np.corrcoef([bbp_test,test_ppa])[0,1]
test_r_sq = corr_test*corr_test

#print('Build R:',corr,'Test R:',corr_test)
#print('Build R^2:',r_sq,'Test R^2:',test_r_sq)

babip_num = np.sum([df_test.loc[:,'1B'],df_test.loc[:,'2B'],df_test.loc[:,'3B']],axis=0)
babip_denom = np.sum([df_test.loc[:,'PA'],df_test.loc[:,'SO']*-1,df_test.loc[:,'BB']*-1,df_test.loc[:,'IBB']*-1],axis=0)
babip = np.divide(babip_num,babip_denom)

luck = np.corrcoef([babip,bbp_test-test_ppa])
#plot the comparison

df_test.loc[:,'PPA'] = test_ppa
df_test.loc[:,'Exp_PPA'] = bbp_test
df_test.loc[:,'PTS'] = np.sum([test_tb,test_discipline,test_speed],axis=0)
df_test.loc[:,'Exp_PTS'] = np.multiply(bbp_test,df_test.loc[:,'PA'])


plt.scatter(df_test.loc[:,'Exp_PPA'],df_test.loc[:,'PPA'])
plt.axis([0.1,0.75,0.1,0.75])
plt.xlabel('Predicted Points per PA')
plt.ylabel('Actual Points per PA')
plt.title('2017 Moneyball Scoring Model, PA>100')
plt.text(0.2, .6, r'$R^2 = 0.84$')
plt.show()

#identify outliers
df_test.loc[:,'Resid'] = test_ppa-bbp_test
outlier =  np.divide(df_test.loc[:,'Resid'],df_test.loc[:,'Exp_PPA'])>0.2
print(sum(outlier))

for i in range(0,len(df_test)):
    if (outlier[i]):
        print(df_test.loc[i,'Name'], df_test.loc[i,'PTS'],df_test.loc[i,'Exp_PTS'] )

outlier =  np.divide(df_test.loc[:,'Resid'],df_test.loc[:,'Exp_PPA'])<-0.2
print(sum(outlier))

for i in range(0,len(df_test)):
    if (outlier[i]):
        print(df_test.loc[i,'Name'], df_test.loc[i,'PTS'],df_test.loc[i,'Exp_PTS'] )

