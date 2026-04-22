
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 18:14:02 2022

@author: PHJT002
"""

import hddm
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from tableone import TableOne, load_dataset
from scipy.special import rel_entr
import random
import numpy as np
########################################################
  
#plotting rts with outliers

#Kernel density estimate plot
def kde_plot(df, conditions, dv, col_name, save_file=False):
    sns.set_style('white')
    sns.set_style('ticks')
    fig, ax = plt.subplots()

    for condition in conditions:
        condition_data = df[(df[col_name] == condition)][dv]
        sns.kdeplot(condition_data, shade=True, label=condition)
        plt.legend()
        
    sns.despine()
    
    if save_file:
        plt.savefig("kernel_density_estimate_seaborn_python_response"
                     "-time.png")
    plt.show()
    
data_rdm_dem=pd.read_csv('data_rdm_dem.csv')

data_rdm_dem['session']=data_rdm_dem['session'].astype('string')
data_rdm_dem['response']=data_rdm_dem['response'].astype('string')
data_rdm_dem['coherence']=data_rdm_dem['coherence'].astype('string')

new_data_rdm_dem = data_rdm_dem[(data_rdm_dem['rt'] > 0.2)] 


new_data_rdm_dem['session']=new_data_rdm_dem['session'].astype('string')


kde_plot(data_rdm_dem, ['0', '1'], 'rt', 'coherence',
         save_file=False)


kde_plot(data_rdm_dem, ['0.15', '0.05'], 'rt', 'coherence',
         save_file=False)

kde_plot(new_data_rdm_dem, ['0', '1'], 'rt', 'response',
         save_file=False)

#HDDM plot
data_dist = hddm.utils.flip_errors(new_data_rdm_dem)

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='RT', ylabel='count', title='RT distributions')
for i, subj_idx in new_data_rdm_dem.groupby('coherence'):
    subj_idx.rt.hist(bins=100, histtype='step', ax=ax)

#z-scoring the variable


age= data_rdm_dem.loc[:,'age']
z_age=stats.zscore(age)
data_rdm_dem['z_age']=z_age

pdi= data_rdm_dem.loc[:,'pdi']
print (pdi)
z_pdi=stats.zscore(pdi)
data_rdm_dem['z_pdi']=z_pdi 

caps=data_rdm_dem.loc[:,'caps']
print (caps)
z_caps=stats.zscore(caps)
data_rdm_dem['z_caps']=z_caps


g='data_rdm_dem.csv'
data_rdm_dem.to_csv(g)



#############################################################
def kl_divergence(a, b):
    return sum(a[i] * np.log(a[i]/b[i]) for i in range(len(a)))
##############################################################

#%%
##Null model
vat= hddm.HDDM(new_data_rdm_dem, p_outlier=0.10)
vat.find_starting_values()
vat.sample(20000, burn=2000, thin=5,  dbname='vat_traces.db', db='pickle')
vat.save('vat')

vat=hddm.load('vat')

vat.print_stats()
vat.plot_posteriors()
vat.plot_posterior_predictive()#
#%%
## Condition  model

vat_coh = hddm.HDDM(new_data_rdm_dem,depends_on={'v': ['coherence'], 
                                                'a':['coherence'],
                                                't':['coherence']}, std_depends=False, p_outlier=0.10)
vat_coh.find_starting_values()
vat_coh.sample(20000, burn=2000, thin=5,  dbname='vat_coh_traces.db', db='pickle')
vat_coh.save('vat_coh')

vat_coh= hddm.load('vat_coh')

vat_coh.print_stats()
vat_coh.plot_posteriors()
vat_coh.plot_posterior_predictive()

#%%
 ## TMS  model

vat_tms=hddm.load('vat_tms')

vat_tms.print_stats()
vat_tms.plot_posteriors()
vat_tms.plot_posterior_predictive()


v_E, v_S = vat_tms.nodes_db.node[['v(1)', 'v(0)']]
hddm.analyze.plot_posterior_nodes([v_E, v_S], bins=20)
plt.legend(['v TMS','v Sham'])
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
print ("P(TMS >Sham)=",(v_E.trace()> v_S.trace()).mean())

a_E, a_S = vat_tms.nodes_db.node[['a(1)', 'a(0)']]
hddm.analyze.plot_posterior_nodes([a_E, a_S], bins=20)
plt.legend(['a TMS','a Sham'])
plt.xlabel('Decision-threshold')
plt.ylabel('Posterior probability')
print ("P(TMS < Sham)=",(a_E.trace()< a_S.trace()).mean())

t_E, t_S = vat_tms.nodes_db.node[['t(1)', 't(0)']]
hddm.analyze.plot_posterior_nodes([t_E, t_S], bins=20)
plt.legend(['t TMS','t Sham'])
plt.xlabel('Non-decision Time')
plt.ylabel('Posterior probability')
print ("P(TMS < Sham)=",(t_E.trace()< t_S.trace()).mean())


vat_tms.plot_posterior_predictive(figsize=(14, 10))



#########
#Get traces of parameters for plotting

vat_tms_traces = vat_tms.get_traces()
vat_tms_traces.to_csv('vat_tms_traces.csv')

v_session=vat_tms_traces[['v(0)','v(1)']].rename(columns={'v(0)':'Sham','v(1)':'TMS'})

a_session=vat_tms_traces[['a(0)','a(1)']].rename(columns={'a(0)':'Sham','a(1)':'TMS'})

t_session=vat_tms_traces[['t(0)','t(1)']].rename(columns={'t(0)':'Sham','t(1)':'TMS'})

ax = sns.violinplot( data=v_session, palette='Spectral_r',linewidth=0.5, inner='point')
ax.set(xlabel ="Session", ylabel = "Drift-rate")  





#v

fig, ax = plt.subplots(figsize=(8, 6))

ax = sns.violinplot( data=v_session, palette='Spectral_r',linewidth=0.5, inner='point')
ax.set(xlabel ="Session", ylabel = "Drift-rate")  

# statistical annotation
x1, x2 = 0, 1   # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_session['Sham'].max() + 0.03, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col)
print ("P(TMS >Sham)=",(v_E.trace()> v_S.trace()).mean())


#a
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.violinplot( data=a_session, palette='Spectral_r',linewidth=0.5, inner='point')
ax.set(xlabel ="Session", ylabel = "Decision-threshold")

# statistical annotation
x1, x2 = 0, 1   # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_session['TMS'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col)
print ("P(TMS < Sham)=",(a_E.trace()< a_S.trace()).mean())


#t
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.violinplot( data=t_session, palette='Spectral_r',linewidth=0.5, inner='point')
ax.set(xlabel ="Session", ylabel = "Non-decision time")

# statistical annotation
x1, x2 = 0, 1   # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = t_session['TMS'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col)
print ("P(TMS < Sham)=",(t_E.trace()< t_S.trace()).mean())



#KL divergence
#We quantified the distance between parameters distributions for Sham and TMS conditions by 
#computing the Kullback-Leibler divergerce between the two distributions. 
 

va_tms_pdi_traces = va_tms_pdi.get_traces()
a_0=vat_tms_traces['a(0.0)']
a_1=vat_tms_traces['a(1.0)']

v_0=vat_tms_traces['v(0.0)']
v_1=vat_tms_traces['v(1.0)']
 
t_0=vat_tms_traces['t(0.0)']
t_1=vat_tms_traces['t(1.0)']
   

KLd_a=kl_divergence(a_0,a_1)
KLd_v=kl_divergence(v_0,v_1)
KLd_t=kl_divergence(t_0,t_1)

    
print('KL-divergence(a_0 || a_1): %.3f ' % kl_divergence(a_0,a_1))  
print('KL-divergence(v_0 || v_1): %.3f ' % kl_divergence(v_0,v_1))
print('KL-divergence(t_0 || t_1): %.3f ' % kl_divergence(t_0,v_1))
  
#check fit

hddm.utils.qp_plot(va_tms, groupby=None, quantiles=(0.1, 0.3, 0.5, 0.7, 0.9), draw_lines=True, ax=None)
va_tms.plot_posterior_predictive()


ppc_data_va_tms= hddm.utils.post_pred_gen(vat_tms_coh, samples=500, append_data=False)
b='va_tms.csv'
va_tms.to_csv(b)

ppc_compare_va_tms= hddm.utils.post_pred_stats(data3, va_tms)
a = 'ppc_compare_mreg_av_PDI1.csv' 
ppc_compare_mreg_av_PDI1.to_csv(a)

#%%
######################################
#model TMS vs Shamx Low Coherence vs High Coherence

data=pd.read_csv('data_rdm_dem.csv')

data=data[data['rt']>0.2]
# data_err = data[data['response'] == 0]
# data_cor = data[data['response'] == 1]

# out_err=data_err['rt'].quantile(.9)
# out_cor=data_cor['rt'].quantile(.9)
# data = data[data['rt'] <= out_err]



vat_tms_coh = hddm.HDDM(data,depends_on={'v': ['coherence','session'], 
                                                'a':['coherence','session'],
                                                't':['coherence','session']}, std_depends=True, p_outlier=0.1)
vat_tms_coh.find_starting_values()
vat_tms_coh.sample(20000, burn=2000, thin=5,  dbname='vat_tms_coh_traces.db', db='pickle')
vat_tms_coh.save('vat_tms_coh')
#%%
vat_tms_coh=hddm.load('vat_tms_coh')
vat_tms_coh.print_stats()
#%%
#model TMS vs Shamx Low PDI vs High PDI

vat_tms_pdi = hddm.HDDM(data,depends_on={'v': ['pdi_group','session'], 
                                                'a':['pdi_group','session'],
                                                't':['pdi_group','session']}, std_depends=True, p_outlier=0.1)
vat_tms_pdi.find_starting_values()
vat_tms_pdi.sample(20000, burn=2000, thin=5,  dbname='vat_tms_pdi_traces2.db', db='pickle')
vat_tms_pdi.save('vat_tms_pdi2')
#%%
vat_tms_pdi.print_stats()


#%%
#model TMS vs Shamx Low CAPS vs High CAPS

vat_tms_caps = hddm.HDDM(data,depends_on={'v': ['caps_group','session'], 
                                                'a':['caps_group','session'],
                                                't':['caps_group','session']}, std_depends=True, p_outlier=0.1)
vat_tms_caps.find_starting_values()
vat_tms_caps.sample(20000, burn=2000, thin=5,  dbname='vat_tms_caps_traces2.db', db='pickle')
vat_tms_caps.save('vat_tms_caps2')
vat_tms_caps.print_stats()



#%%
tms_s=vat_tms_coh.gen_stats()
tms_s=tms_s.T

vat_tms_coh.plot_posteriors()
vat_tms_coh.plot_posterior_predictive()



v_15_0, v_15_1, v_5_0, v_5_1 = vat_tms_coh.nodes_db.node[['v(0.15.0)', 'v(0.15.1)','v(0.05.0)', 'v(0.05.1)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([v_15_0, v_15_1, v_5_0, v_5_1], bins=15)
plt.legend(['v Sham x High Precision','v TMS x High Precision', 'v Sham x Low Precision','v TMS x Low Precision'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for SessionxCondition')
print ("P(v High Precision: Sham >  TMS)=",(v_15_0.trace()< v_15_1.trace()).mean())
print ("P(v Low Precision: Sham >  TMS)=",(v_5_0.trace()< v_5_1.trace()).mean())

a_15_0, a_15_1, a_5_0, a_5_1= vat_tms_coh.nodes_db.node[['a(0.15.0)', 'a(0.15.1)','a(0.05.0)', 'a(0.05.1)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([a_15_0, a_15_1, a_5_0, a_5_1], bins=15)
plt.legend(['a Sham x High Precision','a TMS x High Precision', 'a Sham x Low Precision','a TMS x Low Precision'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for SessionxCondition')
print ("P(a High Precision: Sham > TMS)=",(a_15_0.trace()> a_15_1.trace()).mean())
print ("P(a Low Precision: Sham >  TMS)=",(a_5_0.trace()> a_5_1.trace()).mean())


t_15_0, t_15_1, t_5_0, t_5_1 = vat_tms_coh.nodes_db.node[['t(0.15.0)', 't(0.15.1)','t(0.05.0)', 't(0.05.1)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([t_15_0, t_15_1, t_5_0, t_5_1 ], bins=15)
plt.legend(['t Sham x High Precision','t TMS x High Precision', 't Sham x Low Precision','t TMS x Low Precision'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for SessionxCondition')
print ("P(t High Precision: Sham > TMS)=",(t_15_0.trace()> t_15_1.trace()).mean())
print ("P(t Low Precision: Sham > TMS)=",(t_5_0.trace()> t_5_1.trace()).mean())



vat_tms_coh.plot_posterior_predictive(figsize=(14, 10))


#########
#Get traces of parameters for plotting

vat_tms_coh_traces = vat_tms_coh.get_traces()
vat_tms_coh_traces.to_csv('vat_tms_coh_traces.csv')


v_coh=vat_tms_coh_traces[['v(0.15.0)', 'v(0.15.1)',
                          'v(0.05.0)', 'v(0.05.1)']].rename(columns={'v(0.15.0)':'Sham ','v(0.15.1)':'TMS ',
                                                                                'v(0.05.0)':'Sham','v(0.05.1)':'TMS'})

a_coh=vat_tms_coh_traces[['a(0.15.0)', 'a(0.15.1)',
                          'a(0.05.0)', 'a(0.05.1)']].rename(columns={'a(0.15.0)':'Sham ','a(0.15.1)':'TMS ',
                                                                                'a(0.05.0)':'Sham','a(0.05.1)':'TMS'})

t_coh=vat_tms_coh_traces[['t(0.15.0)', 't(0.15.1)',
                          't(0.05.0)', 't(0.05.1)']].rename(columns={'t(0.15.0)':'Sham ','t(0.15.1)':'TMS ',
                                                                                't(0.05.0)':'Sham','t(0.05.1)':'TMS'})

#Get traces of parameters for KLD

v_0_lpdi=vat_tms_coh_traces['v(low_pdi.0)']
v_1_lpdi=vat_tms_coh_traces['v(low_pdi.1)']
v_0_hpdi=vat_tms_coh_traces['v(high_pdi.0)']
v_1_hpdi=vat_tms_coh_traces['v(high_pdi.1)']

a_0_lpdi=vat_tms_coh_traces['a(low_pdi.0)']
a_1_lpdi=vat_tms_coh_traces['a(low_pdi.1)']
a_0_hpdi=vat_tms_coh_traces['a(high_pdi.0)']
a_1_hpdi=vat_tms_coh_traces['a(high_pdi.1)']

t_0_lpdi=vat_tms_coh_traces['t(low_pdi.0)']
t_1_lpdi=vat_tms_coh_traces['t(low_pdi.1)']
t_0_hpdi=vat_tms_coh_traces['t(high_pdi.0)']
t_1_hpdi=vat_tms_coh_traces['t(high_pdi.1)']
  



#################
#Violin plot

############ v Sham vs TMS

fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.violinplot( data=v_coh, palette='YlGnBu_r',linewidth=0.5, inner='box')
ax.set_xlabel("High Precision                                   Low Precision", size=12, weight="bold",labelpad=20)
ax.set_ylabel("Drift-rate", size=12, weight="bold")

# # statistical annotation
# x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = v_coh['Sham '].max() + 0.05, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col)
# print ("P(v High Precision: Sham <  TMS)=",(v_15_0.trace()< v_15_1.trace()).mean())


# # statistical annotation
# x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = v_coh['Sham'].max() + 0.05, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# print ("P(v Low Precision: Sham <  TMS)=",(v_5_0.trace()< v_5_1.trace()).mean())


# statistical annotation
x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_coh['TMS '].max() + 0.25, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col)
print ("P(v Sham: High Precision <  Low Precision)=",(v_15_0.trace()< v_5_0.trace()).mean())


# statistical annotation
x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_coh['TMS '].max() + 0.3, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col)
print ("P(v TMS: High Precision <  Low Precision)=",(v_15_1.trace()< v_5_1.trace()).mean())



############### a
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.violinplot( data=a_coh, palette='YlGnBu_r',linewidth=0.5, inner='box')
ax.set_xlabel("High Precision                                   Low Precision", size=12, weight="bold",labelpad=20)
ax.set_ylabel("Decision-threshold", size=12, weight="bold")


# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_coh['TMS '].max() + 0.1, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
print ("P(a High Precision: Sham > TMS)=",(a_15_0.trace()> a_15_1.trace()).mean())


# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_coh['TMS'].max() + 0.08, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col)
print ("P(a Low Precision: Sham >  TMS)=",(a_5_0.trace()> a_5_1.trace()).mean())


# statistical annotation
x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_coh['Sham'].max() + 0.15, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col)
print ("P(v Sham: High Precision <  Low Precision)=",(a_15_0.trace()> a_5_0.trace()).mean())


# statistical annotation
x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_coh['TMS'].max() + 0.3, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col,linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col)
print ("P(v TMS: High Precision <  Low Precision)=",(a_15_1.trace()> a_5_1.trace()).mean())


#t
fig, ax = plt.subplots(figsize=(8, 6))
ax = sns.violinplot( data=t_coh, palette='YlGnBu_r',linewidth=0.5, inner='box')
ax.set_xlabel("High Precision                                   Low Precision", size=12, weight="bold",labelpad=20)
ax.set_ylabel("Non-decision Time", size=12, weight="bold")


# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = t_coh['TMS'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col)
print ("P(t High Precision: Sham > TMS)=",(t_15_0.trace()> t_15_1.trace()).mean())

# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = t_coh['TMS '].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col)
print ("P(t Low Precision: Sham > TMS)=",(t_5_0.trace()> t_5_1.trace()).mean())

# statistical annotation
x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = t_coh['Sham '].max() + 0.17, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col)
print ("P(v Sham: High Precision <  Low Precision)=",(t_15_0.trace()< t_5_0.trace()).mean())

# statistical annotation
x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = t_coh['TMS '].max() + 0.14, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col)
print ("P(v TMS: High Precision <  Low Precision)=",(t_15_1.trace()< t_5_1.trace()).mean())


#%%
######################################
#model TMS vs Shamx Low PDI vs High PDI and COHERENCE

vat_tms_pdi_coh = hddm.HDDM(new_data_rdm_dem,depends_on={'v': ['pdi_group','session',"coherence"], 
                                                'a':['pdi_group','session',"coherence"],
                                                't':['pdi_group','session',"coherence"]}, std_depends=True, p_outlier=0.10)
vat_tms_pdi_coh.find_starting_values()
vat_tms_pdi_coh.sample(50000, burn=5000, thin=5,  dbname='vat_tms_pdi_coh_traces.db', db='pickle')
vat_tms_pdi_coh.save('vat_tms_pdi_coh')
#%%
vat_tms_pdi_coh= hddm.load('vat_tms_pdi_coh') 

vat_tms_pdi_coh.print_stats()

vat_tms_pdi_coh.plot_posteriors()
vat_tms_pdi_coh.plot_posterior_predictive()

#%%
v_LPDI_0_LP, v_HPDI_1_LP, v_LPDI_1_LP, v_HPDI_0_LP = vat_tms_pdi_coh.nodes_db.node[['v(0.05.low_pdi.0.0)', 'v(0.05.high_pdi.1.0)','v(0.05.low_pdi.1.0)', 'v(0.05.high_pdi.0.0)']]
v_LPDI_0_HP, v_HPDI_1_HP, v_LPDI_1_HP, v_HPDI_0_HP = vat_tms_pdi_coh.nodes_db.node[['v(0.15.low_pdi.0.0)', 'v(0.15.high_pdi.1.0)','v(0.15.low_pdi.1.0)', 'v(0.15.high_pdi.0.0)']]



PDI_vposteriors= hddm.analyze.plot_posterior_nodes([v_LPDI_0_LP,  v_LPDI_1_LP, v_HPDI_0_LP,v_HPDI_1_LP, ], bins=15)
plt.legend(['v ShamxLPDI','v TMSxLPDI', 'v ShamxHPDI','v TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition in Low Precision')
print ("P(v LPDI: Sham < v TMS)=",(v_LPDI_0_LP.trace()< v_LPDI_1_LP.trace()).mean())
print ("P(v HPDI: Sham < v TMS)=",(v_HPDI_0_LP.trace()< v_HPDI_1_LP.trace()).mean())


PDI_vposteriors= hddm.analyze.plot_posterior_nodes([v_LPDI_0_HP, v_LPDI_1_HP, v_HPDI_0_HP, v_HPDI_1_HP ], bins=15)
plt.legend(['v ShamxLPDI','v TMSxLPDI', 'v ShamxHPDI','v TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition in High precision')
print ("P(v LPDI: Sham < v TMS)=",(v_LPDI_0_HP.trace()< v_LPDI_1_HP.trace()).mean())
print ("P(v HPDI: Sham < v TMS)=",(v_HPDI_0_HP.trace()< v_HPDI_1_HP.trace()).mean())


#%%
a_LPDI_0_LP, a_HPDI_1_LP, a_LPDI_1_LP, a_HPDI_0_LP = vat_tms_pdi_coh.nodes_db.node[['a(0.05.low_pdi.0.0)', 'a(0.05.high_pdi.1.0)','a(0.05.low_pdi.1.0)', 'a(0.05.high_pdi.0.0)']]
a_LPDI_0_HP, a_HPDI_1_HP, a_LPDI_1_HP, a_HPDI_0_HP = vat_tms_pdi_coh.nodes_db.node[['a(0.15.low_pdi.0.0)', 'a(0.15.high_pdi.1.0)','a(0.15.low_pdi.1.0)', 'a(0.15.high_pdi.0.0)']]


PDI_vposteriors= hddm.analyze.plot_posterior_nodes([a_LPDI_0_LP, a_LPDI_1_LP, a_HPDI_0_LP, a_HPDI_1_LP], bins=15)
plt.legend(['a ShamxLPDI','a TMSxLPDI', 'a ShamxHPDI','a TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for PDIxCondition')
print ("P(a LPDI: Sham > a TMS)=",(a_LPDI_0_LP.trace()> a_LPDI_1_LP.trace()).mean())
print ("P(a HPDI: Sham > a TMS)=",(a_HPDI_0_LP.trace()> a_HPDI_1_LP.trace()).mean())


PDI_vposteriors= hddm.analyze.plot_posterior_nodes([a_LPDI_0_HP, a_LPDI_1_HP, a_HPDI_0_HP, a_HPDI_1_HP], bins=15)
plt.legend(['a ShamxLPDI','a TMSxLPDI', 'a ShamxHPDI','a TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for PDIxCondition')
print ("P(a LPDI: Sham > a TMS)=",(a_LPDI_0_HP.trace()> a_LPDI_1_HP.trace()).mean())
print ("P(a HPDI: Sham > a TMS)=",(a_HPDI_0_HP.trace()> a_HPDI_1_HP.trace()).mean())


#%%
t_LPDI_0_LP, t_HPDI_1_LP, t_LPDI_1_LP, t_HPDI_0_LP = vat_tms_pdi_coh.nodes_db.node[['t(0.05.low_pdi.0.0)', 't(0.05.high_pdi.1.0)','t(0.05.low_pdi.1.0)', 't(0.05.high_pdi.0.0)']]
t_LPDI_0_HP, t_HPDI_1_HP, t_LPDI_1_HP, t_HPDI_0_HP = vat_tms_pdi_coh.nodes_db.node[['t(0.15.low_pdi.0.0)', 't(0.15.high_pdi.1.0)','t(0.15.low_pdi.1.0)', 't(0.15.high_pdi.0.0)']]

PDI_vposteriors= hddm.analyze.plot_posterior_nodes([t_LPDI_0_HP, t_LPDI_1_HP, t_HPDI_0_HP, t_HPDI_1_HP], bins=15)
plt.legend(['t ShamxLPDI','t TMSxLPDI', 't ShamxHPDI','t TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for PDIxCondition')
print ("P(t LPDI: Sham > t TMS)=",(t_LPDI_0_HP.trace()> t_LPDI_1_HP.trace()).mean())
print ("P(t HPDI: Sham > t TMS)=",(t_HPDI_0_HP.trace()> t_HPDI_1_HP.trace()).mean())

PDI_vposteriors= hddm.analyze.plot_posterior_nodes([t_LPDI_0_LP, t_LPDI_1_LP, t_HPDI_0_LP, t_HPDI_1_LP], bins=15)
plt.legend(['t ShamxLPDI','t TMSxLPDI', 't ShamxHPDI','t TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for PDIxCondition')
print ("P(t LPDI: Sham > t TMS)=",(t_LPDI_0_LP.trace()> t_LPDI_1_LP.trace()).mean())
print ("P(t HPDI: Sham > t TMS)=",(t_HPDI_0_LP.trace()> t_HPDI_1_LP.trace()).mean())
#%%
######################################
#model TMS vs Shamx Low CAPS vs High CAPS and COHERENCE

vat_tms_caps_coh = hddm.HDDM(new_data_rdm_dem,depends_on={'v': ['caps_group','session',"coherence"], 
                                                'a':['caps_group','session',"coherence"],
                                                't':['caps_group','session',"coherence"]}, std_depends=True, p_outlier=0.10)
vat_tms_caps_coh.find_starting_values()
vat_tms_caps_coh.sample(50000, burn=5000, thin=5,  dbname='vat_tms_caps_coh_traces.db', db='pickle')
vat_tms_caps_coh.save('vat_tms_caps_coh')
#%%
vat_tms_caps_coh= hddm.load('vat_tms_caps_coh') 

vat_tms_caps_coh.print_stats()

vat_tms_caps_coh.plot_posteriors()
vat_tms_caps_coh.plot_posterior_predictive()

#%%
v_LCAPS_0_LP, v_HCAPS_1_LP, v_LCAPS_1_LP, v_HCAPS_0_LP = vat_tms_caps_coh.nodes_db.node[['v(low_caps.0.05.0.0)', 'v(high_caps.0.05.1.0)','v(low_caps.0.05.1.0)', 'v(high_caps.0.05.0.0)']]
v_LCAPS_0_HP, v_HCAPS_1_HP, v_LCAPS_1_HP, v_HCAPS_0_HP = vat_tms_caps_coh.nodes_db.node[['v(low_caps.0.15.0.0)', 'v(high_caps.0.15.1.0)','v(low_caps.0.15.1.0)', 'v(high_caps.0.15.0.0)']]



CAPS_vposteriors= hddm.analyze.plot_posterior_nodes([v_LCAPS_0_LP,  v_LCAPS_1_LP, v_HCAPS_0_LP,v_HCAPS_1_LP, ], bins=15)
plt.legend(['v ShamxLCAPS','v TMSxLCAPS', 'v ShamxHCAPS','v TMSxHCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for CAPSxCondition in Low Precision')
print ("P(v LCAPS: Sham < v TMS)=",(v_LCAPS_0_LP.trace()< v_LCAPS_1_LP.trace()).mean())
print ("P(v HCAPS: Sham < v TMS)=",(v_HCAPS_0_LP.trace()< v_HCAPS_1_LP.trace()).mean())


CAPS_vposteriors= hddm.analyze.plot_posterior_nodes([v_LCAPS_0_HP, v_LCAPS_1_HP, v_HCAPS_0_HP, v_HCAPS_1_HP ], bins=15)
plt.legend(['v ShamxLCAPS','v TMSxLCAPS', 'v ShamxHCAPS','v TMSxHCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for CAPSxCondition in High precision')
print ("P(v LCAPS: Sham < v TMS)=",(v_LCAPS_0_HP.trace()< v_LCAPS_1_HP.trace()).mean())
print ("P(v HCAPS: Sham < v TMS)=",(v_HCAPS_0_HP.trace()< v_HCAPS_1_HP.trace()).mean())


#%%
a_LCAPS_0_LP, a_HCAPS_1_LP, a_LCAPS_1_LP, a_HCAPS_0_LP = vat_tms_caps_coh.nodes_db.node[['a(low_caps.0.05.0.0)', 'a(high_caps.0.05.1.0)','a(low_caps.0.05.1.0)', 'a(high_caps.0.05.0.0)']]
a_LCAPS_0_HP, a_HCAPS_1_HP, a_LCAPS_1_HP, a_HCAPS_0_HP = vat_tms_caps_coh.nodes_db.node[['a(low_caps.0.15.0.0)', 'a(high_caps.0.15.1.0)','a(low_caps.0.15.1.0)', 'a(high_caps.0.15.0.0)']]


CAPS_vposteriors= hddm.analyze.plot_posterior_nodes([a_LCAPS_0_LP, a_LCAPS_1_LP, a_HCAPS_0_LP, a_HCAPS_1_LP], bins=15)
plt.legend(['a ShamxLCAPS','a TMSxLCAPS', 'a ShamxHCAPS','a TMSxHCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for CAPSxCondition')
print ("P(a LCAPS: Sham > a TMS)=",(a_LCAPS_0_LP.trace()> a_LCAPS_1_LP.trace()).mean())
print ("P(a HCAPS: Sham > a TMS)=",(a_HCAPS_0_LP.trace()> a_HCAPS_1_LP.trace()).mean())


CAPS_vposteriors= hddm.analyze.plot_posterior_nodes([a_LCAPS_0_HP, a_LCAPS_1_HP, a_HCAPS_0_HP, a_HCAPS_1_HP], bins=15)
plt.legend(['a ShamxLCAPS','a TMSxLCAPS', 'a ShamxHCAPS','a TMSxHCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for CAPSxCondition')
print ("P(a LCAPS: Sham > a TMS)=",(a_LCAPS_0_HP.trace()> a_LCAPS_1_HP.trace()).mean())
print ("P(a HCAPS: Sham > a TMS)=",(a_HCAPS_0_HP.trace()> a_HCAPS_1_HP.trace()).mean())

print ("P(a TMS: HCAPS > LCAPS)=",(a_HCAPS_1_HP.trace()> a_LCAPS_1_HP.trace()).mean())

#%%
t_LCAPS_0_LP, t_HCAPS_1_LP, t_LCAPS_1_LP, t_HCAPS_0_LP = vat_tms_caps_coh.nodes_db.node[['t(low_caps.0.05.0.0)', 't(high_caps.0.05.1.0)','t(low_caps.0.05.1.0)', 't(high_caps.0.05.0.0)']]
t_LCAPS_0_HP, t_HCAPS_1_HP, t_LCAPS_1_HP, t_HCAPS_0_HP = vat_tms_caps_coh.nodes_db.node[['t(low_caps.0.15.0.0)', 't(high_caps.0.15.1.0)','t(low_caps.0.15.1.0)', 't(high_caps.0.15.0.0)']]

CAPS_vposteriors= hddm.analyze.plot_posterior_nodes([t_LCAPS_0_HP, t_LCAPS_1_HP, t_HCAPS_0_HP, t_HCAPS_1_HP], bins=15)
plt.legend(['t ShamxLCAPS','t TMSxLCAPS', 't ShamxHCAPS','t TMSxHCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for CAPSxCondition')
print ("P(t LCAPS: Sham > t TMS)=",(t_LCAPS_0_HP.trace()> t_LCAPS_1_HP.trace()).mean())
print ("P(t HCAPS: Sham > t TMS)=",(t_HCAPS_0_HP.trace()> t_HCAPS_1_HP.trace()).mean())

CAPS_vposteriors= hddm.analyze.plot_posterior_nodes([t_LCAPS_0_LP, t_LCAPS_1_LP, t_HCAPS_0_LP, t_HCAPS_1_LP], bins=15)
plt.legend(['t ShamxLCAPS','t TMSxLCAPS', 't ShamxHCAPS','t TMSxHCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for CAPSxCondition')
print ("P(t LCAPS: Sham > t TMS)=",(t_LCAPS_0_LP.trace()> t_LCAPS_1_LP.trace()).mean())
print ("P(t HCAPS: Sham > t TMS)=",(t_HCAPS_0_LP.trace()> t_HCAPS_1_LP.trace()).mean())
#%%
######################################
#model TMS vs Shamx Low CAPS vs High PDI

vat_tms_pdi = hddm.HDDM(new_data_rdm_dem,depends_on={'v': ['pdi_group','session'], 
                                                'a':['pdi_group','session'],
                                                't':['pdi_group','session']}, std_depends=True, p_outlier=0.10)
vat_tms_pdi.find_starting_values()
vat_tms_pdi.sample(20000, burn=2000, thin=5,  dbname='vat_tms_pdi_traces.db', db='pickle')
vat_tms_pdi.save('vat_tms_pdi')

vat_tms_pdi= hddm.load('vat_tms_pdi') 

vat_tms_pdi.print_stats()
 
vat_tms_pdi.plot_posteriors()
vat_tms_pdi.plot_posterior_predictive()


v_LPDI_0, v_HPDI_1, v_LPDI_1, v_HPDI_0 = vat_tms_pdi.nodes_db.node[['v(low_pdi.0.0)', 'v(high_pdi.1.0)','v(low_pdi.1.0)', 'v(high_pdi.0.0)']]

PDI_vposteriors= hddm.analyze.plot_posterior_nodes([v_LPDI_0, v_LPDI_1, v_HPDI_0, v_HPDI_1], bins=15)
plt.legend(['v ShamxLPDI','v TMSxLPDI', 'v ShamxHPDI','v TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition')
print ("P(v LPDI: Sham > v TMS)=",(v_LPDI_0.trace()< v_LPDI_1.trace()).mean())
print ("P(v HPDI: Sham > v TMS)=",(v_HPDI_0.trace()< v_HPDI_1.trace()).mean())



a_LPDI_0, a_HPDI_1, a_LPDI_1, a_HPDI_0 = vat_tms_pdi.nodes_db.node[['a(low_pdi.0.0)', 'a(high_pdi.1.0)','a(low_pdi.1.0)', 'a(high_pdi.0.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([a_LPDI_0, a_LPDI_1, a_HPDI_0, a_HPDI_1], bins=15)
plt.legend(['a ShamxLPDI','a TMSxLPDI', 'a ShamxHPDI','a TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for PDIxCondition')
print ("P(a LPDI: Sham > a TMS)=",(a_LPDI_0.trace()> a_LPDI_1.trace()).mean())
print ("P(a HPDI: Sham > a TMS)=",(a_HPDI_0.trace()> a_HPDI_1.trace()).mean())


t_LPDI_0, t_HPDI_1, t_LPDI_1, t_HPDI_0 = vat_tms_pdi.nodes_db.node[['t(low_pdi.0.0)', 't(high_pdi.1.0)','t(low_pdi.1.0)', 't(high_pdi.0.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([t_LPDI_0, t_LPDI_1, t_HPDI_0, t_HPDI_1], bins=15)
plt.legend(['t ShamxLPDI','t TMSxLPDI', 't ShamxHPDI','t TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for PDIxCondition')
print ("P(t LPDI: Sham > t TMS)=",(t_LPDI_0.trace()> t_LPDI_1.trace()).mean())
print ("P(t HPDI: Sham > t TMS)=",(t_HPDI_0.trace()> t_HPDI_1.trace()).mean())
#%%
#########
#Get traces of parameters for plotting

vat_tms_pdi_traces = vat_tms_pdi.get_traces()
vat_tms_pdi_traces.to_csv('vat_tms_pdi_traces.csv')


v_pdi=vat_tms_pdi_traces[['v(low_pdi.0)','v(low_pdi.1)',
                          'v(high_pdi.0)','v(high_pdi.1)']]

a_pdi=vat_tms_pdi_traces[['a(low_pdi.0)','a(low_pdi.1)',
                          'a(high_pdi.0)','a(high_pdi.1)']]

t_pdi=vat_tms_pdi_traces[['t(low_pdi.0)','t(low_pdi.1)',
                          't(high_pdi.0)','t(high_pdi.1)']]

#Get traces of parameters for KLD

v_0_lpdi=vat_tms_pdi_traces['v(low_pdi.0)']
v_1_lpdi=vat_tms_pdi_traces['v(low_pdi.1)']
v_0_hpdi=vat_tms_pdi_traces['v(high_pdi.0)']
v_1_hpdi=vat_tms_pdi_traces['v(high_pdi.1)']

a_0_lpdi=vat_tms_pdi_traces['a(low_pdi.0)']
a_1_lpdi=vat_tms_pdi_traces['a(low_pdi.1)']
a_0_hpdi=vat_tms_pdi_traces['a(high_pdi.0)']
a_1_hpdi=vat_tms_pdi_traces['a(high_pdi.1)']

t_0_lpdi=vat_tms_pdi_traces['t(low_pdi.0)']
t_1_lpdi=vat_tms_pdi_traces['t(low_pdi.1)']
t_0_hpdi=vat_tms_pdi_traces['t(high_pdi.0)']
t_1_hpdi=vat_tms_pdi_traces['t(high_pdi.1)']
  



#################
#Violin plot

############
##v
fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=v_pdi,  palette=sns.diverging_palette(135, 330, s=80, l=70, n=4),linewidth=0.5, inner='box')
ax.set_xlabel('Low PDI                                                 High PDI', size=32, weight="bold",labelpad=80)
ax.set_ylabel("Drift-rate", size=24, weight="bold",labelpad=30)
ax.set_xticklabels(['Sham','TMS','Sham','TMS' ],size=24, weight="bold")


# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_pdi['v(low_pdi.1)'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
print ("P(v LPDI: Sham > v TMS)=",(v_LPDI_0.trace()< v_LPDI_1.trace()).mean())

# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_pdi['v(high_pdi.0)'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col, fontsize=42)
print ("P(v HPDI: Sham > v TMS)=",(v_HPDI_0.trace()< v_HPDI_1.trace()).mean())

# statistical annotation
x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_pdi['v(high_pdi.0)'].max() + 0.1, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "~", ha='center', va='bottom', color=col, fontsize=42)
print ("P(v Sham: High PDI <  Low PDI)=",(v_HPDI_0.trace()< v_LPDI_0.trace()).mean())

# statistical annotation
x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_pdi['v(low_pdi.1)'].max() + 0.3, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, fontsize=42)
print ("P(v TMS: High PDI >  Low PDI)=",(v_HPDI_1.trace()> v_LPDI_1.trace()).mean())


#############
##a
fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=a_pdi,  palette=sns.diverging_palette(135, 330, s=80, l=70, n=4),linewidth=0.5, inner='box')
ax.set_xlabel("Low PDI                                          High PDI", size=32, weight="bold",labelpad=80)
ax.set_ylabel("Decision-threshold", size=24, weight="bold",labelpad=30)
ax.set_xticklabels(['Sham','TMS','Sham','TMS' ],size=24, weight="bold")

# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_pdi['a(low_pdi.1)'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, fontsize=42)
print ("P(a LPDI: Sham > a TMS)=",(a_LPDI_0.trace()> a_LPDI_1.trace()).mean())

# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_pdi['a(high_pdi.1)'].max() + 0.12, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, fontsize=42)
print ("P(a HPDI: Sham > a TMS)=",(a_HPDI_0.trace()> a_HPDI_1.trace()).mean())

# statistical annotation
x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_pdi['a(low_pdi.1)'].max() + 0.2, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
print ("P(v Sham: High PDI <  Low PDI)=",(a_HPDI_0.trace()< a_LPDI_0.trace()).mean())

# statistical annotation
x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_pdi['a(low_pdi.1)'].max() + 0.3, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col, fontsize=42)
print ("P(v TMS: High PDI <  Low PDI)=",(a_HPDI_1.trace()< a_LPDI_1.trace()).mean())


####################
##t
fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=t_pdi, palette='PiYG_r',linewidth=0.5, inner='box')
ax.set_xlabel("Low PDI                                              High PDI", size=32, weight="bold",labelpad=40)
ax.set_ylabel("Non-decision Time", size=24, weight="bold",labelpad=30)
ax.set_xticklabels(['Sham','TMS','Sham','TMS' ],size=24, weight="bold")

# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = t_pdi['t(low_pdi.1)'].max() + 0.03, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col, fontsize=42)
print ("P(a LPDI: Sham > a TMS)=",(t_LPDI_0.trace()> t_LPDI_1.trace()).mean())

# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = t_pdi['t(high_pdi.0)'].max() + 0.03, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col, fontsize=42)
print ("P(a HPDI: Sham > a TMS)=",(t_HPDI_0.trace()> t_HPDI_1.trace()).mean())

# statistical annotation
x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = t_pdi['t(high_pdi.0)'].max() + 0.1, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "~", ha='center', va='bottom', color=col, fontsize=42)
print ("P(v Sham: High PDI <  Low PDI)=",(t_HPDI_0.trace()< t_LPDI_0.trace()).mean())

# statistical annotation
x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = t_pdi['t(high_pdi.0)'].max() + 0.15, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
print ("P(v TMS: High PDI <  Low PDI)=",(t_HPDI_1.trace()< t_LPDI_1.trace()).mean())


#model TMS vs Shamx Low CAPS vs High CAPS

vat_tms_caps = hddm.HDDM(new_data_rdm_dem,depends_on={'v': ['caps_group','session'], 
                                                'a':['caps_group','session'],
                                                't':['caps_group','session']}, std_depends=True, p_outlier=0.10)
vat_tms_caps.find_starting_values()
vat_tms_caps.sample(20000, burn=2000, thin=5,  dbname='vat_tms_caps_traces.db', db='pickle')
vat_tms_caps.save('vat_tms_caps')

vat_tms_caps= hddm.load('vat_tms_caps')


vat_tms_caps.print_stats()
vat_tms_caps.plot_posteriors()
vat_tms_caps.plot_posterior_predictive()

v_LCAPS_0, v_HCAPS_1, v_LCAPS_1, v_HCAPS_0 = vat_tms_caps.nodes_db.node[['v(low_caps.0)', 'v(high_caps.1)','v(low_caps.1)', 'v(high_caps.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([v_LCAPS_0, v_LCAPS_1, v_HCAPS_0, v_HCAPS_1], bins=15)
plt.legend(['v ShamxLCAPS','v TMSxLCAPS', 'v ShamxHCAPS','v TMSxHCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition')
print ("P(v LCAPS: Sham > v TMS)=",(v_LCAPS_0.trace()< v_LCAPS_1.trace()).mean())
print ("P(v HCAPS: Sham > v TMS)=",(v_HCAPS_0.trace()< v_HCAPS_1.trace()).mean())


a_LCAPS_0, a_HCAPS_1, a_LCAPS_1, a_HCAPS_0 = vat_tms_caps.nodes_db.node[['a(low_caps.0)', 'a(high_caps.1)','a(low_caps.1)', 'a(high_caps.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([a_LCAPS_0, a_LCAPS_1, a_HCAPS_0, a_HCAPS_1], bins=15)
plt.legend(['a ShamxLCAPS','a TMSxLCAPS', 'a ShamxHCAPS','a TMSxHCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Decision-threshold')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition')
print ("P(a LCAPS: Sham > a TMS)=",(a_LCAPS_0.trace()> a_LCAPS_1.trace()).mean())
print ("P(a HCAPS: Sham > a TMS)=",(a_HCAPS_0.trace()> a_HCAPS_1.trace()).mean())


t_LCAPS_0, t_HCAPS_1, t_LCAPS_1, t_HCAPS_0 = vat_tms_caps.nodes_db.node[['t(low_caps.0)', 't(high_caps.1)','t(low_caps.1)', 't(high_caps.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([t_LCAPS_0, t_LCAPS_1, t_HCAPS_0, t_HCAPS_1], bins=15)
plt.legend(['t ShamxLCAPS','t TMSxLCAPS', 't ShamxHCAPS','t TMSxHCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Non-decision time')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition')
print ("P(t LCAPS: Sham > t TMS)=",(t_LCAPS_0.trace()> t_LCAPS_1.trace()).mean())
print ("P(t HCAPS: Sham > t TMS)=",(t_HCAPS_0.trace()> t_HCAPS_1.trace()).mean())


#########
#Get traces of parameters for plotting

vat_tms_caps_traces = vat_tms_caps.get_traces()
vat_tms_caps_traces.to_csv('vat_tms_caps_traces.csv')


v_caps=vat_tms_caps_traces[['v(low_caps.0)','v(low_caps.1)',
                          'v(high_caps.0)','v(high_caps.1)']]

a_caps=vat_tms_caps_traces[['a(low_caps.0)','a(low_caps.1)',
                          'a(high_caps.0)','a(high_caps.1)']]

t_caps=vat_tms_caps_traces[['t(low_caps.0)','t(low_caps.1)',
                          't(high_caps.0)','t(high_caps.1)']]


#################
#Violin plot

fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=v_caps, palette=sns.diverging_palette(135, 30, s=80, l=70, n=4),linewidth=0.5, inner='box')
ax.set_xlabel('Low CAPS                                                High CAPS', size=32, weight="bold",labelpad=80)
ax.set_ylabel("Drift-rate", size=24, weight="bold",labelpad=30)
ax.set_xticklabels(['Sham','TMS','Sham','TMS' ],size=24, weight="bold")

# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_caps['v(low_caps.0)'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col, fontsize=42)
print ("P(v LCAPS: Sham < v TMS)=",(v_LCAPS_0.trace()< v_LCAPS_1.trace()).mean())

# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_caps['v(high_caps.0)'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y],  lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col, fontsize=42)
print ("P(v HCAPS: Sham < v TMS)=",(v_HCAPS_0.trace()< v_HCAPS_1.trace()).mean())

# statistical annotation
x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_caps['v(high_caps.0)'].max() + 0.1, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col, fontsize=42)
print ("P(v Sham: High CAPS <  Low CAPS)=",(v_HCAPS_0.trace()< v_LCAPS_0.trace()).mean())

# statistical annotation
x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_caps['v(high_caps.1)'].max() + 0.3, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col, fontsize=42)
print ("P(v TMS: High CAPS <  Low CAPS)=",(v_HCAPS_1.trace()< v_LCAPS_1.trace()).mean())



#a
fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=a_caps,palette=sns.diverging_palette(135, 30, s=80, l=70, n=4),linewidth=0.5, inner='box')
ax.set_xlabel('Low CAPS                                                High CAPS', size=32, weight="bold",labelpad=80)
ax.set_ylabel("Drift-rate", size=24, weight="bold",labelpad=30)
ax.set_xticklabels(['Sham','TMS','Sham','TMS' ],size=24, weight="bold")

# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' CAPS 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_caps['a(low_caps.1)'].max() + 0.1, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col, fontsize=42)
print ("P(a LCAPS: Sham > a TMS)=",(a_LCAPS_0.trace()> a_LCAPS_1.trace()).mean())

# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_caps['a(high_caps.1)'].max() + 0.1, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
print ("P(a HCAPS: Sham > a TMS)=",(a_HCAPS_0.trace()> a_HCAPS_1.trace()).mean())

# statistical annotation
x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_caps['a(low_caps.1)'].max() + 0.5, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
print ("P(v Sham: High CAPS <  Low CAPS)=",(a_HCAPS_0.trace()< a_LCAPS_0.trace()).mean())

# statistical annotation
x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_caps['a(low_caps.1)'].max() + 0.8, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col, fontsize=42)
print ("P(v TMS: High CAPS >  Low CAPS)=",(a_HCAPS_1.trace()> a_LCAPS_1.trace()).mean())


#KL divergence


a_0_lcaps=va_tms_caps_traces['a(low_caps.0.0)']
a_1_lcaps=va_tms_caps_traces['a(low_caps.1.0)']
a_0_hcaps=va_tms_caps_traces['a(high_caps.0.0)']
a_1_hcaps=va_tms_caps_traces['a(high_caps.1.0)']
v_0_lcaps=va_tms_caps_traces['v(low_caps.0.0)']
v_1_lcaps=va_tms_caps_traces['v(low_caps.1.0)']
v_0_hcaps=va_tms_caps_traces['v(high_caps.0.0)']
v_1_hcaps=va_tms_caps_traces['v(high_caps.1.0)']
  

print('KL-divergence(a_0_lcaps || a_1_lcaps): %.3f ' % kl_divergence(a_0_lcaps,a_1_lcaps))
print('KL-divergence(a_0_hcaps || a_1_hcaps): %.3f ' % kl_divergence(a_0_hcaps,a_1_hcaps))
  
print('KL-divergence(v_0_lcaps || v_1_lcaps): %.3f ' % kl_divergence(v_0_lcaps,v_1_lcaps))
print('KL-divergence(v_0_hcaps || v_1_hcaps): %.3f ' % kl_divergence(v_0_hcaps,v_1_hcaps))



#################
#Violin plot for POSTER

############
##v
fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=v_pdi, palette='PiYG_r',linewidth=0.5, inner='point')
ax.set_xlabel('Low PDI     High PDI', size=100, labelpad=100)
ax.set_ylabel("Drift-rate", size=100,labelpad=30)
ax.set_xticklabels(['Sham','TMS','Sham','TMS' ],size=50)


# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_pdi['v(low_pdi.1)'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=10, c=col)
print ("P(v LPDI: Sham > v TMS)=",(v_LPDI_0.trace()< v_LPDI_1.trace()).mean())

# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_pdi['v(high_pdi.0)'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=10, c=col)
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col, fontsize=48)
print ("P(v HPDI: Sham > v TMS)=",(v_HPDI_0.trace()< v_HPDI_1.trace()).mean())

# statistical annotation
x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_pdi['v(high_pdi.0)'].max() + 0.1, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=10, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "~", ha='center', va='bottom', color=col, fontsize=48)
print ("P(v Sham: High PDI <  Low PDI)=",(v_HPDI_0.trace()< v_LPDI_0.trace()).mean())

# statistical annotation
x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_pdi['v(low_pdi.1)'].max() + 0.3, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=10, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, fontsize=48)
print ("P(v TMS: High PDI >  Low PDI)=",(v_HPDI_1.trace()> v_LPDI_1.trace()).mean())


#############
##a
fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=a_pdi, palette='PiYG_r',linewidth=0.5, inner='point')
ax.set_xlabel('Low PDI     High PDI', size=100, labelpad=100)
ax.set_ylabel("Threshold", size=100, labelpad=40)
ax.set_xticklabels(['Sham','TMS','Sham','TMS' ],size=50)

# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_pdi['a(low_pdi.1)'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=10, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, fontsize=48)
print ("P(a LPDI: Sham > a TMS)=",(a_LPDI_0.trace()> a_LPDI_1.trace()).mean())

# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_pdi['a(high_pdi.1)'].max() + 0.12, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=10, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, fontsize=48)
print ("P(a HPDI: Sham > a TMS)=",(a_HPDI_0.trace()> a_HPDI_1.trace()).mean())

# statistical annotation
x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_pdi['a(low_pdi.1)'].max() + 0.2, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=10, c=col, linestyle='dashed')
print ("P(v Sham: High PDI <  Low PDI)=",(a_HPDI_0.trace()< a_LPDI_0.trace()).mean())

# statistical annotation
x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_pdi['a(low_pdi.1)'].max() + 0.3, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=10, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col, fontsize=48)
print ("P(v TMS: High PDI <  Low PDI)=",(a_HPDI_1.trace()< a_LPDI_1.trace()).mean())


################
#KL divergence
#We quantified the distance between parameters distributions for Sham and TMS conditions by 
#computing the Kullback-Leibler divergerce between the two distributions. 
 



KLd_lpdi_a=kl_divergence(a_0_lpdi,a_1_lpdi)
KLd_hpdi_a=kl_divergence(a_0_hpdi,a_1_hpdi)
    
print('KL-divergence(a_0_lpdi || a_1_lpdi): %.3f ' % kl_divergence(a_0_lpdi,a_1_lpdi))
print('KL-divergence(a_0_hpdi || a_1_hpdi): %.3f ' % kl_divergence(a_0_hpdi,a_1_hpdi))
  
print('KL-divergence(v_0_lpdi || v_1_lpdi): %.3f ' % kl_divergence(v_0_lpdi,v_1_lpdi))
print('KL-divergence(v_0_hpdi || v_1_hpdi): %.3f ' % kl_divergence(v_0_hpdi,v_1_hpdi))
  

