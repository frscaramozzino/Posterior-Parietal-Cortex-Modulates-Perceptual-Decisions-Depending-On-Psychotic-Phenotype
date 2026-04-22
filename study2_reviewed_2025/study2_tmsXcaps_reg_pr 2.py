#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:45:45 2023

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

#all 
from plotly.subplots import make_subplots
import plotly.graph_objects as go

#%%

"""TMS CAPS"""


data=pd.read_csv('data_rdm_dem.csv')



data['session']=data['session'].astype('object')
data['coherence']=data['coherence'].astype('object')

data = data[(data['rt'] > 0.2)] 


mreg_avt_tms_CAPS = hddm.load('mreg_avt_tms_CAPS')
mreg_avt_tms_CAPS.print_stats()

statsy=mreg_avt_tms_CAPS.gen_stats()
stats_s=statsy.T


"""
intercept and betas for drift rate
"""
int_v=stats_s.at['mean','v_Intercept']
int_v_sd=stats_s.at['std','v_Intercept']

#TMS
b_v_tms=stats_s.at['mean','v_session[T.1.0]']
bstd_v_tms=stats_s.at['std','v_session[T.1.0]']

#CAPS
b_v_caps=stats_s.at['mean','v_z_caps']
bstd_v_caps=stats_s.at['std','v_z_caps']

#TMSxCAPS
b_v_capsxtms=stats_s.at['mean','v_z_caps:session[T.1.0]']
bstd_v_capsxtms=stats_s.at['std','v_z_caps:session[T.1.0]']

"""
intercept and betas for decision threshold
"""
int_a=stats_s.at['mean','a_Intercept']
int_a_sd=stats_s.at['std','a_Intercept']

#TMS
b_a_tms=stats_s.at['mean','a_session[T.1.0]']
bstd_a_tms=stats_s.at['mean','a_session[T.1.0]']

#CAPS
b_a_caps=stats_s.at['mean','a_z_caps']
bstd_a_caps=stats_s.at['mean','a_z_caps']

#TMSxCAPS
b_a_capsxtms=stats_s.at['mean','a_z_caps:session[T.1.0]']

bstd_a_capsxtms=stats_s.at['std','a_z_caps:session[T.1.0]']

"""
intercept and betas for non-decision time
"""
int_t=stats_s.at['mean','a_Intercept']
int_t_sd=stats_s.at['std','a_Intercept']

#TMS
b_t_tms=stats_s.at['mean','t_session[T.1.0]']
bstd_t_tms=stats_s.at['std','t_session[T.1.0]']

#CAPS
b_t_caps=stats_s.at['mean','t_z_caps']
bstd_t_caps=stats_s.at['std','t_z_caps']

#TMSxCAPS
b_t_capsxtms=stats_s.at['mean','t_z_caps:session[T.1.0]'] 
bstd_t_capsxtms=stats_s.at['std','t_z_caps:session[T.1.0]']


#Generate data

trials_per_level = 11
subjs_per_bin=68



caps=data.caps
caps = (caps - caps.mean()) / caps.std()  # z-score the x factor



for x in caps:
    
    a = int_a+b_a_caps*x  
    v = int_v+b_v_caps*x  
    t = int_t+b_t_caps*x  

    a_tms = int_a+b_a_tms+b_a_caps*x  
    v_tms = int_v+b_v_tms+b_v_caps*x  
    t_tms = int_t+b_t_tms+b_t_caps*x   
    
    a_tmsxcaps = int_a +b_a_capsxtms*x
    v_tmsxcaps = int_v+b_v_capsxtms*x
    t_tmsxcaps = int_t+b_t_capsxtms*x
    
    a_std = int_a_sd+bstd_a_caps*x 
    v_std = int_v_sd+bstd_v_caps*x  
    t_std = int_t_sd+bstd_t_caps*x  
    
    a_tms_std = int_a_sd +bstd_a_tms+bstd_a_caps*x  
    v_tms_std = int_v_sd+bstd_v_tms+bstd_v_caps*x  
    t_tms_std = int_t_sd+bstd_t_tms+bstd_t_caps*x 
      
    a_capsxtms_std = int_a_sd +bstd_a_capsxtms*x
    v_capsxtms_std = int_v_sd+bstd_v_capsxtms*x
    t_capsxtms_std = int_t_sd+bstd_t_capsxtms*x
    

level_sham1 = {'v':v.mean(),'a':a.mean(), 't':t.mean(),
                  'sv':v_std.mean(), 
                  'sa':a_std.mean(), 
                  "st":t_std.mean(),"p":0.1}

level_sham2 = {'v':v.mean(),'a':a.mean(), 't':t.mean(),
                  'sv':v_std.mean(), 
                  'sa':a_std.mean(), 
                  "st":t_std.mean(),"p":0.1}

level_tms = {'v':v_tms_std.mean(),'a':a_tms.mean(), 't':t_tms.mean(),
                   'sv':v_tms_std.mean(), 
                   'sa':a_tms_std.mean(), 
                   "st":t_tms_std.mean(),"p":0.1} 


level_tmsxcaps = {'v':a_tmsxcaps.mean()  , 'a':a_tmsxcaps.mean(), 't':t_tmsxcaps.mean(),
                  'sv':v_capsxtms_std.mean(), 
                  'sa':a_capsxtms_std.mean(), 
                  "st":t_capsxtms_std.mean(),"p":0.1} 

# note that for subjs_per_bin > 1, these are just the mean values of the parameters; indiv subjs within bin are sampled from distributions with the given means, but can still differ within bin around those means. 
#not including sv, sz, st in the statement ensures those are actually 0.

data_sim, params_sim = hddm.generate.gen_rand_data({'level_sham1': level_sham1,
                                                    'level_sham2': level_sham2,
                                                    'level_tms': level_tms,
                                                    'level_tmsxcaps': level_tmsxcaps}, 
                                                       size=trials_per_level, 
                                                       subjs=subjs_per_bin)
#%%
####plot actual and simulated rt

data.loc[data['response'] == 1, 'response'] = 'Correct'
data.loc[data['response'] == 0, 'response'] = 'Incorrect'
data_sim.loc[data_sim['response'] == 1, 'response'] = 'Correct'
data_sim.loc[data_sim['response'] == 0, 'response'] = 'Incorrect'

rt_act=data[['rt','response']]
rt_act=rt_act.assign(rt_type = "observed")

rt_sim=data_sim.assign(rt_type = "simulated")

comp_rt = pd.concat([rt_act, rt_sim])



fig = px.histogram(comp_rt, x="rt", color='rt_type',
                   histnorm='density',
                   facet_row='response',
                   orientation='v',
                   labels={'rt': 'Reaction time',
                            'response': 'Response',
                            'rt_type': 'Source'},
                   barmode='overlay',
                   width=1733,
                   height=900)


fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# Update yaxis properties

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=22,
    title="RT Null-model",
    title_font_color="black",
    legend_title_font_color="blue",
    legend_font_size=22
)
fig.update_xaxes(title_font_family="Arial")

fig.show()
#%%
###plot observed and simulated quantiles in qqplot
from quantile_simulation_function import get_mean_sim_quantiles
from quantile_simulation_function import get_quantiles
sample=range(100)
simulated_data = []

trials_per_level = 11
subjs_per_bin=68


for i in sample:
    
    data_sim, params_sim = hddm.generate.gen_rand_data({'level_sham1': level_sham1,
                                                        'level_sham2': level_sham2,
                                                        'level_tms': level_tms,
                                                        'level_tmsxcaps': level_tmsxcaps}, 
                                                           size=trials_per_level, 
                                                           subjs=subjs_per_bin)
        

    # data_sim.loc[data_sim['condition'] == 'level_sham', 'session'] = '0'
    # data_sim.loc[data_sim['condition'] == 'level_tmsxcaps', 'session'] = '1'

    
    # data_sim=data_sim[['rt','response','session']]

    data_sim= data_sim[(data_sim['rt'] > 0.2)].assign(rt_type = "simulated")

    simulated_data.append(data_sim)
    pd.concat(simulated_data).to_csv('simulated_data_capsreg.csv')

mean_simulated_quantiles_tmsxcaps=get_mean_sim_quantiles(simulated_data)

mean_simulated_quantiles_tmsxcaps=mean_simulated_quantiles_tmsxcaps.assign(source = "simulated")
mean_simulated_quantiles_tmsxcaps.to_csv('mean_simulated_quantiles_tmsxcaps.csv')

#%%
mean_simulated_quantiles_tmsxcaps=get_mean_sim_quantiles(simulated_data)

mean_simulated_quantiles_tmsxcaps=mean_simulated_quantiles_tmsxcaps.assign(source = "simulated")
mean_simulated_quantiles_tmsxcaps.to_csv('mean_simulated_quantiles_tmsxcaps.csv')

data=data[data['rt']>0.2]

observed_quantiles=get_quantiles(data).assign(source = "observed")

#%%

from sklearn.metrics import mean_squared_error
mse='MSE:'

mse_1 = str(mean_squared_error(observed_quantiles['q_correct'], mean_simulated_quantiles_tmsxcaps['q_correct']).round(3))
mse_1="".join([mse,mse_1])
mse_0 = str(mean_squared_error(observed_quantiles['q_incorrect'], mean_simulated_quantiles_tmsxcaps['q_incorrect']).round(3))
mse_0="".join([mse,mse_0])
print(mse_1)
print(mse_0)
size_font=40
#qq plots
fig = make_subplots(
    rows=1, cols=2
)

fig.add_trace(go.Scatter(
    y=mean_simulated_quantiles_tmsxcaps['q_correct'],
    x=observed_quantiles['q_correct'],
    name='Correct'),
col=1, row=1)

fig.add_trace(go.Scatter(
    y=mean_simulated_quantiles_tmsxcaps['q_incorrect'],
    x=observed_quantiles['q_incorrect'],
    name='Incorrect'),
col=2, row=1)

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    name="Optimal simulation",
    mode="lines",
    line=dict(color="Black")
))
fig.add_shape(type='line',
                x0=0,
                y0=0,
                x1=observed_quantiles['q_correct'].max(),
                y1=observed_quantiles['q_correct'].max(),
                line=dict(color='black'),
                row=1,
                col=1)
fig.add_shape(type='line',
                x0=0,
                y0=0,
                x1=observed_quantiles['q_incorrect'].max(),
                y1=observed_quantiles['q_incorrect'].max(),
                line=dict(color='black'),
                row=1,
                col=2)

fig.add_annotation(
       y=5
     , x=3
     , text=mse_1
     ,showarrow=False
     , font=dict(size=22, color="black", family="Courier New"),
    col=1, row=1)
fig.add_annotation(
       y=5
     , x=3
     , text=mse_0
     ,showarrow=False
     , font=dict(size=22, color="black", family="Courier New"),
    col=2, row=1)

# Update xaxis properties
fig.update_xaxes(title_text="Observed RT quantiles", range=[0,6])

# Update yaxis properties
fig.update_yaxes(title_text="Simulated RT quantiles",range=[0,10])

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=size_font, 
    title_font_color="black",   
    title='A. HDDM-LR-TMSxCAPS',    
    legend_title_font_color="blue",
    legend_font_size=size_font
)

fig.show()
#%%

#simulating CAPS  values by randomly sampling from our data

caps_sam=data.caps.dropna()
f=pd.Series(caps_sam.sample(n=68,replace=True)).reset_index()
caps_sim=pd.concat([f]*40, ignore_index=True)#to have a value of CAPS for each trial
caps_sim=caps_sim['caps']
data_sim['z_caps']=(caps_sim - caps_sim.mean())/caps_sim.std(ddof=0)

data_sim=data_sim[['rt','response','condition','z_caps']]

data_sim.loc[data_sim['condition'] == 'level_sham1', 'session'] = '0'
data_sim.loc[data_sim['condition'] == 'level_sham2', 'session'] = '0'
data_sim.loc[data_sim['condition'] == 'level_tms', 'session'] = '1'
data_sim.loc[data_sim['condition'] == 'level_tmsxcaps', 'session'] = '1'



data_sim=data_sim[['rt','response','session','z_caps']].dropna()
data_sim= data_sim[(data_sim['rt'] > 0.2)].assign(rt_type = "simulated")
#%%

sim_mreg_avt_tms_CAPS= hddm.models.HDDMRegressor(data_sim, ['a ~ 1+z_caps+session+z_caps*session', 
                                                           'v ~ 1++z_caps+session+z_caps*session', 
                                                           't ~ 1++z_caps+session+z_caps*session'],
                                                group_only_regressors=True,
                                                p_outlier=0.10)
sim_mreg_avt_tms_CAPS.find_starting_values()
sim_mreg_avt_tms_CAPS.sample(20000, burn=2000,thin=5, dbname='sim_mreg_avt_tms_CAPS_traces.db', db='pickle')
sim_mreg_avt_tms_CAPS.save('sim_mreg_avt_tms_CAPS')

sim_mreg_avt_tms_CAPS = hddm.load('sim_mreg_avt_tms_CAPS')

sim_mreg_avt_tms_CAPS.print_stats()

