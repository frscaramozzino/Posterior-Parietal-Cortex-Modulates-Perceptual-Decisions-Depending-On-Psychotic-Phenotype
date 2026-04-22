# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:17:21 2022

@author: PHJT002
"""
import pandas as pd
import numpy as np
import scipy.stats as stats
import glob
from functools import reduce
import re
from tableone import TableOne, load_dataset
import matplotlib.pyplot as plt
#%%

data= pd.read_csv('dem_pdi_caps.csv')

data_rdm_dem=pd.read_csv('data_rdm_dem.csv')
data_bt_dem=pd.read_csv('data_bt_dem.csv')
data_srt_dem=pd.read_csv('data_srt_dem.csv')

#%%
data_rdm_dem=pd.read_csv('data_rdm_dem.csv')

data_dem= groupby[]

res = stats.spearmanr([1, 2, 3, 4, 5], [5, 6, 7, 8, 7])
res.statistic
0.8207826816681233
res.pvalue
0.08858700531354381

#%%
#confidence
data_rdm_dem_sub=data_rdm_dem.groupby(['participant'])['session','gender','age','pdi', 'caps', 'rt', 'response','confidence'].mean().dropna()
data_rdm_dem_sub['session']=data_rdm_dem_sub['session'].astype('string')
data_rdm_dem_sub['gender']=data_rdm_dem_sub['gender'].astype('string')



data_bt_dem_sub=data_bt_dem.groupby(['participant'])['session','gender','draws','pdi', 'caps','confidence_BT'].mean()
data_bt_sub=data_bt.groupby(['participant'])['draws'].mean()

data_srt_dem_sub=data_srt_dem.groupby(['participant'])['session','gender','age','pdi', 'caps', 'rt', 'response'].median()


m_caps=data_rdm_dem.caps.median()
m_pdi=data_rdm_dem.pdi.median()

data_rdm_dem_sub['pdi_group'] = np.where(data_rdm_dem_sub['pdi']>m_pdi, 'high_pdi', 'low_pdi')
data_rdm_dem_sub['caps_group'] = np.where(data_rdm_dem_sub['caps']>m_caps, 'high_caps', 'low_caps')



data_bt_dem_sub['pdi_group'] = np.where(data_rdm_dem_sub['pdi']>m_pdi, 'high_pdi', 'low_pdi')
data_bt_dem_sub['caps_group'] = np.where(data_rdm_dem_sub['caps']>m_caps, 'high_caps', 'low_caps')

data[['pdi','caps']].dropna().plot.kde(figsize=[12,8])
plt.legend(['Age (years)', 'PDI', 'CAPS'])
plt.xlim([-30,250])

data[['age','pdi','caps']].boxplot(whis=3)
plt.show()



#TMS vs Sham dem
# columns to summarize
columns = [ 'age', 'gender', 'pdi',"caps"]



# non-normal variables
nonnormal = ['gender', 'pdi',"caps"]

 
# optionally, a categorical variable for stratification
groupby = ['session']



table = TableOne(data_rdm_dem_sub, columns=columns, groupby=groupby,
                  nonnormal=nonnormal, label_suffix=True,pval = True, smd=True,
                  htest_name=True)
            

table

table.to_excel('table_session.xlsx')

print(table_RDM.tabulate(tablefmt="github"))


#TPDI groups dem
# columns to summarize
columns = [ 'age', 'gender', 'pdi',"caps"]



# non-normal variables
nonnormal = ['gender', 'pdi',"caps"]

 
# optionally, a categorical variable for stratification
groupby = ['pdi_group']



table = TableOne(data_rdm_dem_sub, columns=columns, groupby=groupby,
                  nonnormal=nonnormal, label_suffix=True,pval = True, smd=True,
                  htest_name=True)
            

table

table.to_excel('table_pdi.xlsx')

print(table_RDM.tabulate(tablefmt="github"))


#CAPS dem
# columns to summarize
columns = [ 'age', 'gender', 'pdi',"caps"]



# non-normal variables
nonnormal = ['gender', 'pdi',"caps"]

 
# optionally, a categorical variable for stratification
groupby = ['caps_group']



table = TableOne(data_rdm_dem_sub, columns=columns, groupby=groupby,
                  nonnormal=nonnormal, label_suffix=True,pval = True, smd=True,
                  htest_name=True)
            

table

table.to_excel('table_caps.xlsx')

print(table_RDM.tabulate(tablefmt="github"))


#TMS vs Sham RDM
# columns to summarize
columns = [ 'rt', 'response', 'confidence',"age"]



# non-normal variables
nonnormal = ['rt', 'response', 'confidence',"age"]

 
# optionally, a categorical variable for stratification
groupby = ['pdi_group']



table_RDM = TableOne(data_rdm_dem_sub, columns=columns, groupby=groupby,
                  nonnormal=nonnormal, label_suffix=True,pval = True, smd=True,
                  htest_name=True)
            

table_RDM
print(table_RDM.tabulate(tablefmt="github"))


#TMS vs Sham BT

columns = [ 'draws']



# non-normal variables
nonnormal = ['draws']

 
# optionally, a categorical variable for stratification
groupby = ['session']



table_bt_dem = TableOne(data_bt_dem_sub, columns=columns, groupby=groupby,
                  nonnormal=nonnormal, label_suffix=True,pval = True, smd=True,
                  htest_name=True)
table_bt_dem




data_bt_dem[['draws']].dropna().plot.kde(figsize=[12,8])
plt.legend(['DTD'])
plt.xlim([-10,30])




data_bt_dem_sub=data_bt_dem.groupby(['participant'])['session','gender','age','pdi', 'caps', 'draws', 'acc','confidence_BT'].mean()


m_caps=caps.median()
m_pdi=pdi.median()

data_bt_dem_sub['pdi_group'] = np.where(data_bt_dem_sub['pdi']>m_pdi, 'high_pdi', 'low_pdi')
data_bt_dem_sub['caps_group'] = np.where(data_bt_dem_sub['caps']>m_caps, 'high_caps', 'low_caps')



# columns to summarize
columns = [ 'draws', 'acc', 'confidence_BT']

# non-normal variables
nonnormal = [ 'gender', 'age', 'pdi', 'caps', 'acc','draws',]


# optionally, a categorical variable for stratification
groupby = ['session']



table_BT= TableOne(data_bt_dem_sub, columns=columns, groupby=groupby,
                  nonnormal=nonnormal, label_suffix=True,pval = True, smd=True,
                  htest_name=True)
            

table_BT




#TMS vs Sham SRT
# columns to summarize
columns = [ 'rt']


# non-normal variables
nonnormal = ['rt']


# optionally, a categorical variable for stratification
groupby = ['session']



table_SRT = TableOne(data_srt_dem_sub, columns=columns, groupby=groupby,
                  nonnormal=nonnormal, label_suffix=True,pval = True, smd=True,
                  htest_name=True)
            

table_SRT



#Low PDI vs High PDI RDM
# columns to summarize
columns = [ 'rt', 'response', 'confidence']



# non-normal variables
nonnormal = ['rt', 'response', 'confidence']


# optionally, a categorical variable for stratification
groupby = ['pdi_group']



table_RDM_PDI = TableOne(data_rdm_dem, columns=columns, groupby=groupby,
                  nonnormal=nonnormal, label_suffix=True,pval = True, smd=True,
                  htest_name=True)
            

table_RDM_PDI


#Low PDI vs High PDI BT
# columns to summarize
columns = [ 'draws', 'acc', 'confidence_BT']

# non-normal variables
nonnormal = ['draws', 'acc', 'confidence_BT']


# optionally, a categorical variable for stratification
groupby = ['pdi_group']



table_BT_PDI= TableOne(data_bt_dem, columns=columns, groupby=groupby,
                  nonnormal=nonnormal, label_suffix=True,pval = True, smd=True,
                  htest_name=True)
            

table_BT_PDI

#Low PDI vs High PDI SRT
# columns to summarize
columns = [ 'rt', 'response']


# non-normal variables
nonnormal = ['rt', 'response']


# optionally, a categorical variable for stratification
groupby = ['pdi_group']



table_SRT_PDI = TableOne(data_srt_dem, columns=columns, groupby=groupby,
                  nonnormal=nonnormal, label_suffix=True,pval = True, smd=True,
                  htest_name=True)
            

table_SRT_PDI
