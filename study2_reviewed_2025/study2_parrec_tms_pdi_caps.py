#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:06:24 2023

@author: francescoscaramozzino
"""

import hddm
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import statistics
import seaborn as sns
from functools import reduce
import re
from tableone import TableOne, load_dataset

import numpy as np


#####################
#potting with plotly
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'



data_rdm_dem=pd.read_csv('data_rdm_dem.csv')

data_rdm_dem['session']=data_rdm_dem['session'].astype('object')
data_rdm_dem['coherence']=data_rdm_dem['coherence'].astype('object')

data_rdm_dem = data_rdm_dem[(data_rdm_dem['rt'] > 0.2)] 
out = data_rdm_dem[(data_rdm_dem['rt'] > 6)] 


data_rdm_dem['z_pdi']=(data_rdm_dem.pdi - data_rdm_dem.pdi.mean())/data_rdm_dem.pdi.std(ddof=0)
data_rdm_dem['z_caps']=(data_rdm_dem.caps - data_rdm_dem.caps.mean())/data_rdm_dem.caps.std(ddof=0)

data_rdm_dem.dropna()

mreg_avt_tms_PDI = hddm.load('mreg_avt_tms_PDI')


mreg_avt_tms_PDI.print_stats()
statsy=mreg_avt_tms_PDI.gen_stats()
stats_s=statsy.T

mreg_avt_tms_PDI_traces=mreg_avt_tms_PDI.get_traces()


###drift rate
int_v=stats_s.at['mean','v_Intercept']
int_v_sd=stats_s.at['std','v_Intercept']
 
#PDI
b_v_pdi=stats_s.at['mean','v_z_pdi']


#tms
b_v_tms=stats_s.at['mean','v_session[T.1.0]']

#tmsxpdi
b_v_tmsxpdi=stats_s.at['mean','v_z_pdi:session[T.1.0]']

###decision threshold
int_a=stats_s.at['mean','a_Intercept']

#PDI
b_a_pdi=stats_s.at['mean','a_z_pdi']

#tms

b_a_tms=stats_s.at['mean','a_session[T.1.0]']

#tmsxpdi
b_a_tmsxpdi=stats_s.at['mean','a_z_pdi:session[T.1.0]']

###t
int_t=stats_s.at['mean','t_Intercept']

#PDI
b_t_pdi=stats_s.at['mean','t_z_pdi']

#tms
b_t_tms=stats_s.at['mean','t_session[T.1.0]']

#tmsxpdi
b_t_tmsxpdi=stats_s.at['mean','t_z_pdi:session[T.1.0]']


pdi=data_rdm_dem.pdi



#Generate data

trials_per_level = 20
subjs_per_bin=68

for x in pdi:
    xx = (pdi - pdi.mean()) / pdi.std()  # z-score the x factor
    
    a = int_a+b_a_pdi*xx+b_a_tmsxpdi*xx   #  indiv subj param values that are centered on intercept but deviate from it up or down by z-scored x
    v = int_v+b_v_pdi*xx+b_v_tmsxpdi*xx    # can also do for drift, here using same beta coeff
    t = int_t+b_t_pdi*xx+b_t_tmsxpdi*xx    # can also do for drift, here using same beta coeff
    
    a1 = int_a+b_a_pdi*xx+b_a_tmsxpdi*xx  
    v1 = int_v+b_v_tms+b_v_pdi*xx+b_v_tmsxpdi*xx    
    t1 = int_t+b_t_tms+b_t_pdi*xx+b_t_tmsxpdi*xx  
    
  
parvec_cond_0 = {'v':v.mean()  , 'a':a.mean()  , 't':t.mean()} 
parvec_cond_1 = {'v':v1.mean()  , 'a':a1.mean()  , 't':t1.mean()} # set a to value set by regression, here v is set to constant

# note that for subjs_per_bin > 1, these are just the mean values of the parameters; indiv subjs within bin are sampled from distributions with the given means, but can still differ within bin around those means. 
#not including sv, sz, st in the statement ensures those are actually 0.

data_sim, params_sim = hddm.generate.gen_rand_data({'level_cond_0': parvec_cond_0,
                                                        'level_cond_1': parvec_cond_1}, 
                                                       size=trials_per_level, 
                                                       subjs=subjs_per_bin)
    
#simulating PDI  values by randomly sampling from our data

pdi_sam=data_rdm_dem.pdi.dropna()
f=pd.Series(pdi_sam.sample(n=68,replace=True)).reset_index()
pdi_sim=pd.concat([f]*40, ignore_index=True)#to have a value of CAPS for each trial
pdi_sim=pdi_sim['pdi']
data_sim['z_pdi']=(pdi_sim - pdi_sim.mean())/pdi_sim.std(ddof=0)

data_sim=data_sim[['rt','response','condition','z_pdi']]

data_sim.loc[data_sim['condition'] == 'level_cond_0', 'session'] = '0'
data_sim.loc[data_sim['condition'] == 'level_cond_1', 'session'] = '1'

data_sim=data_sim[['rt','response','session','z_pdi']].dropna()

data_sim = data_sim[(data_sim['rt'] > 0.2)] 

#plot actual and simulated rt
rt_act=data_rdm_dem[['rt','response']]
rt_act=rt_act.assign(rt_type = "actual")

rt_sim=data_sim[['rt','response']]
rt_sim=data_sim.assign(rt_type = "simulated")

comp_rt=pd.concat([rt_act, rt_sim])

fig = px.histogram(comp_rt, x="rt", color='rt_type',  
                histnorm='density',
                facet_row='response',
                orientation='v',
                opacity=0.4,
                barmode='overlay',
                width=1733,
                height=900)


fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=20, 
    title_font_family="Times New Roman",
    title="A.",
    title_font_color="black",    legend_title_font_color="blue",
    legend_font_size=22
)
fig.update_xaxes(title_font_family="Arial")


sim_mreg_avt_tms_PDI= hddm.models.HDDMRegressor(data_sim, ['a ~ 1+z_pdi+session+z_pdi*session', 
                                                           'v ~ 1++z_pdi+session+z_pdi*session', 
                                                           't ~ 1++z_pdi+session+z_pdi*session'],
                                                group_only_regressors=True,
                                                p_outlier=0.10)
sim_mreg_avt_tms_PDI.find_starting_values()
sim_mreg_avt_tms_PDI.sample(20000, burn=2000,thin=5, dbname='sim_mreg_avt_tms_PDI_traces.db', db='pickle')
sim_mreg_avt_tms_PDI.save('sim_mreg_avt_tms_PDI')

sim_mreg_avt_tms_PDI = hddm.load('sim_mreg_avt_tms_PDI')

sim_mreg_avt_tms_PDI.print_stats()



a_PDI= sim_mreg_avt_tms_PDI.nodes_db.node ['a_z_pdi'] 
hddm.analyze.plot_posterior_nodes ([a_PDI], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (a_PDI.trace() > 0).mean())

v_PDI = sim_mreg_avt_tms_PDI.nodes_db.node ['v_z_pdi'] 
hddm.analyze.plot_posterior_nodes ([v_PDI], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (v_PDI.trace() < 0).mean())

t_PDI = sim_mreg_avt_tms_PDI.nodes_db.node ['t_z_pdi'] 
hddm.analyze.plot_posterior_nodes ([t_PDI], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (t_PDI.trace() < 0).mean())



a_tms= sim_mreg_avt_tms_PDI.nodes_db.node ['a_session[T.1]'] 
hddm.analyze.plot_posterior_nodes ([a_tms], bins=8)
plt.legend(['β for TMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (a_tms.trace() > 0).mean())

v_tms= sim_mreg_avt_tms_PDI.nodes_db.node ['v_session[T.1]'] 
hddm.analyze.plot_posterior_nodes ([v_tms], bins=8)
plt.legend(['β for TMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI > 0) = ", (v_tms.trace() >0).mean())

t_tms= sim_mreg_avt_tms_PDI.nodes_db.node ['t_session[T.1]'] 
hddm.analyze.plot_posterior_nodes ([t_tms], bins=8)
plt.legend(['β for TMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (t_tms.trace() < 0).mean())



a_PDI_tms= sim_mreg_avt_tms_PDI.nodes_db.node ['a_z_pdi:session[T.1]'] 
hddm.analyze.plot_posterior_nodes ([a_PDI_tms], bins=8)
plt.legend(['β for PDIxTMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (a_PDI_tms.trace() > 0).mean())

v_PDI_tms = sim_mreg_avt_tms_PDI.nodes_db.node ['v_z_pdi:session[T.1]'] 
hddm.analyze.plot_posterior_nodes ([v_PDI_tms], bins=8)
plt.legend(['β for PDIxTMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (v_PDI_tms.trace() < 0).mean())

t_PDI_tms = sim_mreg_avt_tms_PDI.nodes_db.node ['t_z_pdi:session[T.1]'] 
hddm.analyze.plot_posterior_nodes ([t_PDI_tms], bins=8)
plt.legend(['β for PDIxTMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (t_PDI_tms.trace() > 0).mean())


