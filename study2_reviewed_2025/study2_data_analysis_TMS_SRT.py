# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:41:39 2022

@author: PHJT002
"""

import hddm
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from tableone import TableOne, load_dataset

import statsmodels.formula.api as smf


data_srt_dem= hddm.load_csv('data_srt_dem.csv').dropna()
# data_srt_dem['session']=data_srt_dem['session'].astype('string')
data_srt_dem['response']=data_srt_dem['response'].astype('string')


rt= data_srt_dem.loc[:,'rt']
z_rt=stats.zscore(rt)
data_srt_dem['z_rt']=z_rt
#%%

#HDDM plot
data_dist = hddm.utils.flip_errors(data_srt_dem)

fig = plt.figure()
ax = fig.add_subplot(111, xlabel='RT', ylabel='count', title='RT distributions')
for i, subj_idx in data_srt_dem.groupby(['session']):
    subj_idx.rt.hist(bins=100, histtype='step', ax=ax)

plt.savefig('hddm_demo_fig_00.pdf')
#%%

#SRT
############
##rt RDM ~TMS
model_rt_SRT= smf.mixedlm("rt ~  1+session+position+session*position",
                    data_srt_dem,
                    groups= data_srt_dem['participant'], 
                    )

mdf_rt_SRT= model_rt_SRT.fit(method=["lbfgs"])
print(mdf_rt_SRT.summary())

#%%
#z-scoring the variable


age= data_srt_dem.loc[:,'age']
z_age=stats.zscore(age)
data_srt_dem['z_age']=z_age

pdi= data_srt_dem.loc[:,'pdi']
print (pdi)
z_pdi=stats.zscore(pdi)
data_srt_dem['z_pdi']=z_pdi 

caps=data_srt_dem.loc[:,'caps']
print (caps)
z_caps=stats.zscore(caps)
data_srt_dem['z_caps']=z_caps




#response encoded SRT model
pos_srt_data=data_srt_dem[(data_srt_dem['rt'] > 0.2)]
pos_srt_data.loc[pos_srt_data["position"] == "a", "position"] = 0
pos_srt_data.loc[pos_srt_data["position"] == "s", "position"] = 1
pos_srt_data[['position']] = pos_srt_data[['position']].apply(pd.to_numeric) 
pos_srt_data.position.apply(pd.to_numeric)

pos_srt_data=pos_srt_data.rename(columns = {"response": "accuracy"})
pos_srt_data=pos_srt_data.rename(columns = {"position": "response"})


#model TMS vs Sham
p_vat_tms_srt = hddm.HDDM(pos_srt_data,depends_on={'v': 'session', 'a':'session','t':'session'}, std_depends=True, p_outlier=0.10)
p_vat_tms_srt.find_starting_values()
p_vat_tms_srt.sample(20000, burn=2000, thin=5,  dbname='p_vat_tms_srt_traces.db', db='pickle')
p_vat_tms_srt.save('p_vat_tms_srt')

p_vat_tms_srt= hddm.load('p_vat_tms_srt')

p_vat_tms_srt.print_stats()

p_vat_tms_srt.plot_posteriors()

p_vat_tms_srt.plot_posterior_predictive()

 
v_E, v_S = p_vat_tms_srt.nodes_db.node[['v(1.0)', 'v(0.0)']]
hddm.analyze.plot_posterior_nodes([v_E, v_S], bins=20)
plt.legend(['v TMS','v Sham'])
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
print ("P(TMS >Sham)=",(v_E.trace()> v_S.trace()).mean())

a_E, a_S = p_vat_tms_srt.nodes_db.node[['a(1.0)', 'a(0.0)']]
hddm.analyze.plot_posterior_nodes([a_E, a_S], bins=20, )
plt.legend(['a TMS','a Sham'])
plt.xlabel('Decision-threshold')
plt.ylabel('Posterior probability')
print ("P(TMS > Sham)=",(a_E.trace()> a_S.trace()).mean())

t_E, t_S = p_vat_tms_srt.nodes_db.node[['t(1.0)', 't(0.0)']]
hddm.analyze.plot_posterior_nodes([t_E, t_S], bins=20)
plt.legend(['t TMS','t Sham'])
plt.xlabel('Non-decision time')
plt.ylabel('Posterior probability')
print ("P(TMS < Sham)=",(t_E.trace()< t_S.trace()).mean())


####################################################################


#model TMS vs Sham
vat_srt = hddm.HDDM(pos_srt_data, p_outlier=0.10)
vat_srt.find_starting_values()
vat_srt.sample(20000, burn=2000, thin=5,  dbname='vat_srt_traces.db', db='pickle')
vat_srt.save('vat_srt')

vat_srt= hddm.load('vat_srt')

vat_srt.print_stats()
vat_srt.plot_posterior_predictive()

#model TMS vs Sham
vat_tms_srt = hddm.HDDM(pos_srt_data,depends_on={'v': 'session', 'a':'session','t':'session'}, std_depends=True, p_outlier=0.10)
vat_tms_srt.find_starting_values()
vat_tms_srt.sample(20000, burn=2000, thin=5,  dbname='vat_tms_srt_traces.db', db='pickle')
vat_tms_srt.save('vat_tms_srt')



vat_tms_srt= hddm.load('vat_tms_srt')

vat_tms_srt.print_stats()

vat_tms_srt.plot_posteriors()

vat_tms_srt.plot_posterior_predictive()

 
v_E, v_S = vat_tms_srt.nodes_db.node[['v(1.0)', 'v(0.0)']]
hddm.analyze.plot_posterior_nodes([v_E, v_S], bins=20)
plt.legend(['v TMS','v Sham'])
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
print ("P(TMS >Sham)=",(v_E.trace()> v_S.trace()).mean())

a_E, a_S = vat_tms_srt.nodes_db.node[['a(1.0)', 'a(0.0)']]
hddm.analyze.plot_posterior_nodes([a_E, a_S], bins=20, )
plt.legend(['a TMS','a Sham'])
plt.xlabel('Decision-threshold')
plt.ylabel('Posterior probability')
print ("P(TMS > Sham)=",(a_E.trace()> a_S.trace()).mean())

t_E, t_S = vat_tms_srt.nodes_db.node[['t(1.0)', 't(0.0)']]
hddm.analyze.plot_posterior_nodes([t_E, t_S], bins=20)
plt.legend(['t TMS','t Sham'])
plt.xlabel('Non-decision time')
plt.ylabel('Posterior probability')
print ("P(TMS < Sham)=",(t_E.trace()< t_S.trace()).mean())



#########
#Get traces of parameters for plotting

vat_tms_srt_traces = vat_tms_srt.get_traces()

v_tms=vat_tms_srt_traces[['v(0.0)','v(1.0)']].rename(columns={'v(0.0)':'Sham','v(1.0)':'TMS'})

a_tms=vat_tms_srt_traces[['a(0.0)','a(1.0)']].rename(columns={'a(0.0)':'Sham','a(1.0)':'TMS'})

t_tms=vat_tms_srt_traces[['t(0.0)','t(1.0)']].rename(columns={'t(0.0)':'Sham','t(1.0)':'TMS'})

                                                                            
                                                                                

#################
#Violin plot

############
##v
fig, ax = plt.subplots(figsize=(20, 20))
ax = sns.violinplot( data=v_tms, palette='YlGnBu_r',linewidth=0.5, inner='box')
ax.set_ylabel("Drift rate", size=36, weight="bold",  labelpad=40)
ax.set_xticklabels(v_tms, fontsize=36)
ax.tick_params(axis='y', labelsize=36)
# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_tms['Sham'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col)
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col, fontsize=45)
print ("P(TMS >Sham)=",(v_E.trace()> v_S.trace()).mean())

#############
##a
fig, ax = plt.subplots(figsize=(20, 20))
ax = sns.violinplot( data=a_tms, palette='YlGnBu_r',linewidth=0.5, inner='point')
ax.set_ylabel("Decision-threshold",  size=36, weight="bold",  labelpad=40)
ax.set_xticklabels(a_tms, fontsize=36)
ax.tick_params(axis='y', labelsize=36)
# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_tms['Sham'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, fontsize=45)
print ("P(TMS > Sham)=",(a_E.trace()>a_S.trace()).mean())

####################
##t
fig, ax = plt.subplots(figsize=(20, 20))
ax = sns.violinplot( data=t_tms, palette='YlGnBu_r',linewidth=0.5, inner='point')
ax.set_ylabel("Non-decision Time",  size=36, weight="bold",  labelpad=40)
ax.set_xticklabels(t_tms, fontsize=36)
ax.tick_params(axis='y', labelsize=36)
# statistical annotation
# x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = t_tms['TMS'].max() + 0.03, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col)
# plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col, fontsize=45)
# print ("P(TMS < Sham)=",(t_E.trace()>t_S.trace()).mean())

                                                                                
                     
                                                                                
                                                                                
                                                                                
                                                                                
#model TMS vs Sham| PDI groups
vat_tms_pdi_srt = hddm.HDDM(pos_srt_data,depends_on={'v': ['pdi_group','session'], 
                                                'a':['pdi_group','session'],
                                                't':['pdi_group','session']}, std_depends=False, p_outlier=0.05)
vat_tms_pdi_srt.find_starting_values()
vat_tms_pdi_srt.sample(100000, burn=15000, thin=5,  dbname='vat_tms_pdi_srt_traces.db', db='pickle')
vat_tms_pdi_srt.save('vat_tms_pdi_srt')


vat_tms_caps_srt = hddm.HDDM(pos_srt_data,depends_on={'v': ['caps_group','session'], 
                                                'a':['caps_group','session'],
                                                't':['caps_group','session']}, std_depends=False, p_outlier=0.05)
vat_tms_caps_srt.find_starting_values()
vat_tms_caps_srt.sample(10000, burn=15000, thin=5,  dbname='vat_tms_caps_srt_traces.db', db='pickle')
vat_tms_caps_srt.save('vat_tms_caps_srt')




vat_tms_pdi_srt= hddm.load('vat_tms_pdi_srt')

vat_tms_pdi_srt.print_stats()

vat_tms_pdi_srt.plot_posteriors()
vat_tms_srt.plot_posterior_predictive()


v_LPDI_0, v_HPDI_1, v_LPDI_1, v_HPDI_0 = vat_tms_pdi_srt.nodes_db.node[['v(low_pdi.0.0)', 'v(high_pdi.1.0)','v(low_pdi.1.0)', 'v(high_pdi.0.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([v_LPDI_0, v_LPDI_1, v_HPDI_0, v_HPDI_1], bins=15)
plt.legend(['v ShamxLPDI','v TMSxLPDI', 'v ShamxHPDI','v TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition')
print ("P(v LPDI: Sham > v TMS)=",(v_LPDI_0.trace()< v_LPDI_1.trace()).mean())
print ("P(v HPDI: Sham > v TMS)=",(v_HPDI_0.trace()< v_HPDI_1.trace()).mean())
print ("P(v Sham: HPDI > v LPDI)=",(v_LPDI_0.trace()< v_HPDI_0.trace()).mean())
print ("P(v TMS: HPDI > v LPDI)=",(v_LPDI_1.trace()< v_HPDI_1.trace()).mean())



a_LPDI_0, a_HPDI_1, a_LPDI_1, a_HPDI_0 = vat_tms_pdi_srt.nodes_db.node[['a(low_pdi.0.0)', 'a(high_pdi.1.0)','a(low_pdi.1.0)', 'a(high_pdi.0.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([a_LPDI_0, a_LPDI_1, a_HPDI_0, a_HPDI_1], bins=15)
plt.legend(['a ShamxLPDI','a TMSxLPDI', 'a ShamxHPDI','a TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for PDIxCondition')
print ("P(a LPDI: Sham > a TMS)=",(a_LPDI_0.trace()> a_LPDI_1.trace()).mean())
print ("P(a HPDI: Sham > a TMS)=",(a_HPDI_0.trace()> a_HPDI_1.trace()).mean())
print ("P(a Sham: HPDI > v LPDI)=",(a_LPDI_0.trace()< a_HPDI_0.trace()).mean())
print ("P(a TMS: HPDI > v LPDI)=",(a_LPDI_1.trace()< a_HPDI_1.trace()).mean())


t_LPDI_0, t_HPDI_1, t_LPDI_1, t_HPDI_0 = vat_tms_pdi_srt.nodes_db.node[['t(low_pdi.0.0)', 't(high_pdi.1.0)','t(low_pdi.1.0)', 't(high_pdi.0.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([t_LPDI_0, t_LPDI_1, t_HPDI_0, t_HPDI_1], bins=15)
plt.legend(['t ShamxLPDI','t TMSxLPDI', 't ShamxHPDI','t TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for PDIxCondition')
print ("P(t LPDI: Sham > t TMS)=",(t_LPDI_0.trace()> t_LPDI_1.trace()).mean())
print ("P(t HPDI: Sham > t TMS)=",(t_HPDI_0.trace()> t_HPDI_1.trace()).mean())
print ("P(t Sham: HPDI > v LPDI)=",(t_LPDI_0.trace()< t_HPDI_0.trace()).mean())
print ("P(t TMS: HPDI > v LPDI)=",(t_LPDI_1.trace()< t_HPDI_1.trace()).mean())

#########
#Get traces of parameters for plotting

vat_tms_pdi_srt_traces = vat_tms_pdi_srt.get_traces()


v_pdi=vat_tms_pdi_srt_traces[['v(low_pdi.0.0)','v(low_pdi.1.0)',
                          'v(high_pdi.0.0)','v(high_pdi.1.0)']]
                                                                                

a_pdi=vat_tms_pdi_srt_traces[['a(low_pdi.0.0)','a(low_pdi.1.0)',
                          'a(high_pdi.0.0)','a(high_pdi.1.0)']]  

t_pdi=vat_tms_pdi_srt_traces[['t(low_pdi.0.0)','t(low_pdi.1.0)',
                          't(high_pdi.0.0)','t(high_pdi.1.0)']]

#Get traces of parameters for KLD

v_0_lpdi=vat_tms_pdi_srt_traces['v(low_pdi.0.0)']
v_1_lpdi=vat_tms_pdi_srt_traces['v(low_pdi.1.0)']
v_0_hpdi=vat_tms_pdi_srt_traces['v(high_pdi.0.0)']
v_1_hpdi=vat_tms_pdi_srt_traces['v(high_pdi.1.0)']

a_0_lpdi=vat_tms_pdi_srt_traces['a(low_pdi.0.0)']
a_1_lpdi=vat_tms_pdi_srt_traces['a(low_pdi.1.0)']
a_0_hpdi=vat_tms_pdi_srt_traces['a(high_pdi.0.0)']
a_1_hpdi=vat_tms_pdi_srt_traces['a(high_pdi.1.0)']

t_0_lpdi=vat_tms_pdi_srt_traces['t(low_pdi.0.0)']
t_1_lpdi=vat_tms_pdi_srt_traces['t(low_pdi.1.0)']
t_0_hpdi=vat_tms_pdi_srt_traces['t(high_pdi.0.0)']
t_1_hpdi=vat_tms_pdi_srt_traces['t(high_pdi.1.0)']
  


############
##v
fig, ax = plt.subplots(figsize=(20, 20))
ax = sns.violinplot( data=v_pdi, palette=sns.diverging_palette(130, 330, s=100, l=70, n=4),linewidth=0.5, inner='box')
ax.set_xlabel("Low PDI                              High PDI", size=36, weight="bold",labelpad=40)
ax.set_ylabel("Drift rate", size=36, weight="bold", labelpad=40)
ax.set_xticklabels(['Sham','TMS','Sham','TMS' ],size=36)
ax.tick_params(axis='y', labelsize=36)

# # statistical annotation
# x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = v_pdi['Sham '].max() + 0.05, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# print ("P(v LPDI: Sham > v TMS)=",(v_LPDI_0.trace()< v_LPDI_1.trace()).mean())

# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_pdi['v(high_pdi.0.0)'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col)
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col,fontsize=45)

# # statistical annotation
# x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = v_pdi['v(high_pdi.0.0)'].max() + 0.1, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col, linestyle='dashed')
# plt.text((x1+x2)*.5, y+h, "~", ha='center', va='bottom', color=col,fontsize=45)

# statistical annotation
x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_pdi['v(high_pdi.0.0)'].max() + 0.3, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col,fontsize=45)


#############
##a
fig, ax = plt.subplots(figsize=(20, 20))
ax = sns.violinplot( data=a_pdi, palette=sns.diverging_palette(130, 330, s=100, l=70, n=4),linewidth=0.5, inner='box')
ax.set_xlabel("Low PDI                              High PDI", size=36, weight="bold",labelpad=40)
ax.set_ylabel("Decision threshold", size=36, weight="bold", labelpad=40)
ax.set_xticklabels(['Sham','TMS','Sham','TMS' ],size=36)
ax.tick_params(axis='y', labelsize=36)

# # statistical annotation
# x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = a_pdi['a(low_pdi.1.0)'].max() + 0.12, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col)
# plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col,fontsize=45)

# # statistical annotation
# x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = a_pdi['a(high_pdi.1.0)'].max() + 0.12, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col)
# plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col,fontsize=45)

# # statistical annotation
# x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = a_pdi['a(high_pdi.0)'].max() + 0.2, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
# print ("P(v Sham: High PDI <  Low PDI)=",(a_HPDI_0.trace()< a_LPDI_0.trace()).mean())

# # statistical annotation
# x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = a_pdi['a(high_pdi.1)'].max() + 0.2, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
# plt.text((x1+x2)*.5, y+h, "~", ha='center', va='bottom', color=col,fontsize=32)
# print ("P(v TMS: High PDI <  Low PDI)=",(a_HPDI_1.trace()< a_LPDI_1.trace()).mean())


####################
##t
fig, ax = plt.subplots(figsize=(20, 20))
ax = sns.violinplot( data=t_pdi, palette=sns.diverging_palette(130, 330, s=100, l=70, n=4),linewidth=0.5, inner='box')
ax.set_xlabel("Low PDI                              High PDI", size=36, weight="bold",labelpad=40)
ax.set_ylabel("Non-decision time", size=36, weight="bold", labelpad=40)
ax.set_xticklabels(['Sham','TMS','Sham','TMS' ],size=36)
ax.tick_params(axis='y', labelsize=36)

# # statistical annotation
# x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = t_pdi['t(low_pdi.1)'].max() + 0.03, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col,fontsize=32)
# print ("P(a LPDI: Sham > a TMS)=",(t_LPDI_0.trace()> t_LPDI_1.trace()).mean())

# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = t_pdi['t(high_pdi.1.0)'].max() + 0.01, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col)
plt.text((x1+x2)*.5, y+h, "~", ha='center', va='bottom', color=col,fontsize=45)

# statistical annotation
x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = t_pdi['t(high_pdi.0.0)'].max() + 0.03, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col,fontsize=45)

# # statistical annotation
# x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = t_pdi['TMS'].max() + 0.1, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
# print ("P(v TMS: High PDI <  Low PDI)=",(t_HPDI_1.trace()< t_LPDI_1.trace()).mean())




#model TMS vs Sham| CAPS groups
vat_tms_caps_srt = hddm.HDDM(new_data_srt_dem,depends_on={'v': ['caps_group','session'], 
                                                'a':['caps_group','session'],
                                                't':['caps_group','session']}, std_depends=False, p_outlier=0.05)
vat_tms_caps_srt.find_starting_values()
vat_tms_caps_srt.sample(100000, burn=15000, thin=5,  dbname='vat_tms_srt_traces.db', db='pickle')
vat_tms_caps_srt.save('vat_tms_caps_srt')

vat_tms_caps_srt= hddm.load('vat_tms_caps_srt')

vat_tms_caps_srt.print_stats()

vat_tms_caps_srt.plot_posteriors()
vat_tms_caps_srt.plot_posterior_predictive()


v_LCAPS_0, v_HCAPS_1, v_LCAPS_1, v_HCAPS_0 = vat_tms_caps_srt.nodes_db.node[['v(low_caps.0.0)', 'v(high_caps.1.0)','v(low_caps.1.0)', 'v(high_caps.0.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([v_LCAPS_0, v_LCAPS_1, v_HCAPS_0, v_HCAPS_1], bins=15)
plt.legend(['v ShamxLCAPS','v TMSxLCAPS', 'v ShamxHCAPS','v TMSxHCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition')
print ("P(v LCAPS: Sham > v TMS)=",(v_LCAPS_0.trace()< v_LCAPS_1.trace()).mean())
print ("P(v HCAPS: Sham > v TMS)=",(v_HCAPS_0.trace()< v_HCAPS_1.trace()).mean())
print ("P(v Sham: HCAPS > v LCAPS)=",(v_LCAPS_0.trace()< v_HCAPS_0.trace()).mean())
print ("P(v TMS: HCAPS > v LCAPS)=",(v_LCAPS_1.trace()< v_HCAPS_1.trace()).mean())



a_LCAPS_0, a_HCAPS_1, a_LCAPS_1, a_HCAPS_0 = vat_tms_caps_srt.nodes_db.node[['a(low_caps.0.0)', 'a(high_caps.1.0)','a(low_caps.1.0)', 'a(high_caps.0.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([a_LCAPS_0, a_LCAPS_1, a_HCAPS_0, a_HCAPS_1], bins=15)
plt.legend(['a ShamxLCAPS','a TMSxLCAPS', 'a ShamxHCAPS','a TMSxHCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Decision-threshold')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition')
print ("P(a LCAPS: Sham > a TMS)=",(a_LCAPS_0.trace()> a_LCAPS_1.trace()).mean())
print ("P(a HCAPS: Sham > a TMS)=",(a_HCAPS_0.trace()> a_HCAPS_1.trace()).mean())
print ("P(a Sham: HCAPS > v LCAPS)=",(a_LCAPS_0.trace()< a_HCAPS_0.trace()).mean())
print ("P(a TMS: HCAPS > v LCAPS)=",(a_LCAPS_1.trace()< a_HCAPS_1.trace()).mean())


t_LCAPS_0, t_HCAPS_1, t_LCAPS_1, t_HCAPS_0 = vat_tms_caps_srt.nodes_db.node[['t(low_caps.0.0)', 't(high_caps.1.0)','t(low_caps.1.0)', 't(high_caps.0.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([t_LCAPS_0, t_LCAPS_1, t_HCAPS_0, t_HCAPS_1], bins=15)
plt.legend(['t ShamxLCAPS','t TMSxLCAPS', 't ShamxHCAPS','t TMSxHCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Non-decision time')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition')
print ("P(t LCAPS: Sham > t TMS)=",(t_LCAPS_0.trace()> t_LCAPS_1.trace()).mean())
print ("P(t HCAPS: Sham > t TMS)=",(t_HCAPS_0.trace()> t_HCAPS_1.trace()).mean())
print ("P(t Sham: HCAPS > v LCAPS)=",(t_LCAPS_0.trace()< t_HCAPS_0.trace()).mean())
print ("P(t TMS: HCAPS > v LCAPS)=",(t_LCAPS_1.trace()< t_HCAPS_1.trace()).mean())


#########
#Get traces of parameters for plotting

vat_tms_caps_srt_traces = vat_tms_caps_srt.get_traces()

v_caps=vat_tms_caps_srt_traces[['v(low_caps.0.0)','v(low_caps.1.0)',
                          'v(high_caps.0.0)','v(high_caps.1.0)']]

a_caps=vat_tms_caps_srt_traces[['a(low_caps.0.0)','a(low_caps.1.0)',
                          'a(high_caps.0.0)','a(high_caps.1.0)']]

t_caps=vat_tms_caps_srt_traces[['t(low_caps.0.0)','t(low_caps.1.0)',
                          't(high_caps.0.0)','t(high_caps.1.0)']]


#################
#Violin plot


#################
#Violin plot

#v
fig, ax = plt.subplots(figsize=(20, 20))
ax = sns.violinplot( data=v_caps, palette=sns.diverging_palette(130, 33, s=100, l=70, n=4),linewidth=0.5, inner='box')
ax.set_xlabel("Low CAPS                           High CAPS", size=36, weight="bold",labelpad=40)
ax.set_ylabel("Drift rate", size=36, weight="bold", labelpad=40)
ax.set_xticklabels(['Sham','TMS','Sham','TMS' ],size=36)
ax.tick_params(axis='y', labelsize=36)

# statistical annotation
x1, x2 = 0, 1  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_caps['v(low_caps.0.0)'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col)
plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col,fontsize=45)

# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = v_caps['v(high_caps.0.0)'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col)
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col,fontsize=45)

# # statistical annotation
# x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = v_caps['Sham'].max() + 0.1, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
# plt.text((x1+x2)*.5, y+h, "", ha='center', va='bottom', color=col,fontsize=32)
# print ("P(v Sham: High CAPS <  Low CAPS)=",(v_HCAPS_0.trace()< v_LCAPS_0.trace()).mean())

# # statistical annotation
# x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = v_caps['TMS'].max() + 0.3, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
# plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col,fontsize=32)
# print ("P(v TMS: High CAPS <  Low CAPS)=",(v_HCAPS_1.trace()< v_LCAPS_1.trace()).mean())



#a
fig, ax = plt.subplots(figsize=(20, 20))
ax = sns.violinplot( data=a_caps, palette=sns.diverging_palette(130, 33, s=100, l=70, n=4),linewidth=0.5, inner='box')
ax.set_xlabel("Low CAPS                           High CAPS", size=36, weight="bold",labelpad=40)
ax.set_ylabel("Decision threshold", size=36, weight="bold", labelpad=40)
ax.set_xticklabels(['Sham','TMS','Sham','TMS' ],size=36)
ax.tick_params(axis='y', labelsize=36)


# # statistical annotation
# x1, x2 = 0, 1  # columns 'Sham' CAPS 'TMS' (first column: 0, see plt.xticks())
# y, h, col = a_caps['a(low_caps.1)'].max() + 0.08, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col)
# plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col,fontsize=45)

# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_caps['a(high_caps.0.0)'].max() + 0.05, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col)
plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col,fontsize=45)

# statistical annotation
x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_caps['a(high_caps.0.0)'].max() + 0.2, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col,fontsize=45)

# statistical annotation
x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = a_caps['a(high_caps.0.0)'].max() + 0.4, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "~", ha='center', va='bottom', color=col,fontsize=45)


#t
fig, ax = plt.subplots(figsize=(20, 20))
ax = sns.violinplot( data=t_caps, palette=sns.diverging_palette(130, 33, s=100, l=70, n=4),linewidth=0.5, inner='box')
ax.set_xlabel("Low CAPS                           High CAPS", size=36, weight="bold",labelpad=40)
ax.set_ylabel("Non-decision time", size=36, weight="bold", labelpad=40)
ax.set_xticklabels(['Sham','TMS','Sham','TMS' ],size=36)
ax.tick_params(axis='y', labelsize=36)




# x1, x2 = 0, 1  # columns 'Sham' CAPS 'TMS' (first column: 0, see plt.xticks())
# y, h, col = t_caps['TMS '].max() + 0.05, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
# plt.text((x1+x2)*.5, y+h, "***", ha='center', va='bottom', color=col,fontsize=32)
# print ("P(t LCAPS: Sham > t TMS)=",(t_LCAPS_0.trace()> t_LCAPS_1.trace()).mean())

# statistical annotation
x1, x2 = 2, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = t_caps['t(high_caps.1.0)'].max() + 0.02, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col,fontsize=45)

# statistical annotation
x1, x2 = 0, 2  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
y, h, col = t_caps['t(low_caps.0.0)'].max() + 0.03, 0.01, 'k'
plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=2.5, c=col, linestyle='dashed')
plt.text((x1+x2)*.5, y+h, "**", ha='center', va='bottom', color=col,fontsize=45)

# # statistical annotation
# x1, x2 = 1, 3  # columns 'Sham' and 'TMS' (first column: 0, see plt.xticks())
# y, h, col = t_caps['TMS'].max() + 0.25, 0.01, 'k'
# plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col, linestyle='dashed')
# plt.text((x1+x2)*.5, y+h, "~", ha='center', va='bottom', color=col,fontsize=32)
# print ("P(v TMS: High CAPS <  Low CAPS)=",(t_LCAPS_1.trace()< t_HCAPS_1.trace()).mean())

