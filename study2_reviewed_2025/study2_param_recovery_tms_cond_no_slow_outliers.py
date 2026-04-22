#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 17:30:33 2022

@author: francescoscaramozzino
"""


import hddm
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import scipy.stats as stats
import numpy as np


#####################
#potting with plotly
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'


from plotly.subplots import make_subplots
import plotly.graph_objects as go


#Create simulated data for parameter recovery
###plot observed and simulated quantiles in qqplot
from quantile_simulation_function import get_mean_sim_quantiles
from quantile_simulation_function import get_condition_quantiles
from quantile_simulation_function import get_quantiles


data=pd.read_csv('data_rdm_dem.csv')

data=data[data['rt']>0.2]


#Create simulated data for parameter recovery

vat_tms_coh= hddm.load('vat_tms_coh')
vat_tms_coh.print_stats()


trials_per_level = 20
subjs_per_bin=34


tms_s=vat_tms_coh.gen_stats()
tms_s=tms_s.T


#Set up parameters by conditions from previously run model
level_sham_15 = {'v':tms_s.at['mean','v(0.15.0.0)'],
                 'a':tms_s.at['mean','a(0.15.0.0)'], 
                 "t":tms_s.at['mean','t(0.15.0.0)'], 
                 'sv':tms_s.at['std','v(0.15.0.0)'], 
                 'sa':tms_s.at['std','a(0.15.0.0)'], 
                 "st":tms_s.at['std','t(0.15.0.0)'],
                 "p":0.1}

level_sham_05 = {'v':tms_s.at['mean','v(0.05.0.0)'],
                 'a':tms_s.at['mean','a(0.05.0.0)'], 
                 "t":tms_s.at['mean','t(0.05.0.0)'], 
                 'sv':tms_s.at['std','v(0.05.0.0)'], 
                 'sa':tms_s.at['std','a(0.05.0.0)'], 
                 "st":tms_s.at['std','t(0.05.0.0)'],
                                  "p":0.1}


level_tms_15 = {'v':tms_s.at['mean','v(0.15.1.0)'],
                 'a':tms_s.at['mean','a(0.15.1.0)'], 
                 "t":tms_s.at['mean','t(0.15.1.0)'], 
                 'sv':tms_s.at['std','v(0.15.1.0)'], 
                 'sa':tms_s.at['std','a(0.15.1.0)'], 
                 "st":tms_s.at['std','t(0.15.1.0)'],
                                  "p":0.1}


level_tms_05 = {'v':tms_s.at['mean','v(0.05.1.0)'],
                 'a':tms_s.at['mean','a(0.05.1.0)'], 
                 "t":tms_s.at['mean','t(0.05.1.0)'], 
                 'sv':tms_s.at['std','v(0.05.1.0)'], 
                 'sa':tms_s.at['std','a(0.05.1.0)'], 
                 "st":tms_s.at['std','t(0.05.1.0)'],
                                  "p":0.1}


sample=range(100)
simulated_data = []


for i in sample:
    data_sim, params = hddm.generate.gen_rand_data({'level_sham_15':level_sham_15,
                                                  'level_sham_05':level_sham_05,
                                             'level_tms_15':level_tms_15,
                                                  'level_tms_05':level_tms_05},
                                                  size = trials_per_level,
                                                  subjs=subjs_per_bin)
    
    
    data_sim.loc[data_sim['condition'] == 'level_sham_15', 'cond'] = '0.15'
    data_sim.loc[data_sim['condition'] == 'level_tms_15', 'cond'] = '0.15'
    data_sim.loc[data_sim['condition'] == 'level_sham_05', 'cond'] = '0.05'
    data_sim.loc[data_sim['condition'] == 'level_tms_05', 'cond'] = '0.05'
    
    data_sim.loc[data_sim['condition'] == 'level_sham_15', 'session'] = '0'
    data_sim.loc[data_sim['condition'] == 'level_sham_05', 'session'] = '0'
    data_sim.loc[data_sim['condition'] == 'level_tms_15', 'session'] = '1'
    data_sim.loc[data_sim['condition'] == 'level_tms_05', 'session'] = '1'
        
    data_sim=data_sim[['rt','response','session','cond']]

    data_sim= data_sim[(data_sim['rt'] > 0.2)].assign(rt_type = "simulated")

    simulated_data.append(data_sim)
    pd.concat(simulated_data).to_csv('simulated_data_coherence_tms.csv')

mean_simulated_quantiles_coherence_tms=get_mean_sim_quantiles(simulated_data)
mean_simulated_quantiles_coherence_tms=mean_simulated_quantiles_coherence_tms.assign(source = "simulated")

mean_simulated_quantiles_coherence_tms.to_csv('mean_simulated_quantiles_coherence_tms_noout.csv')



#%%
# compute the mean squared error

from sklearn.metrics import mean_squared_error

data=pd.read_csv('data_rdm_dem.csv')

data=data[data['rt']>0.2]

mean_simulated_quantiles_coherence_tms=pd.read_csv('mean_simulated_quantiles_coherence_tms_noout.csv')

observed_quantiles=get_quantiles(data)

mse='MSE:'

mse_1 = str(mean_squared_error(observed_quantiles['q_correct'], mean_simulated_quantiles_coherence_tms['q_correct']).round(3))
mse_1="".join([mse,mse_1])
mse_0 = str(mean_squared_error(observed_quantiles['q_incorrect'], mean_simulated_quantiles_coherence_tms['q_incorrect']).round(3))
mse_0="".join([mse,mse_0])
print(mse_1)
print(mse_0)
size_font=40
#qq plots
fig = make_subplots(
    rows=1, cols=2
)

fig.add_trace(go.Scatter(
    y=mean_simulated_quantiles_coherence_tms['q_correct'],
    x=observed_quantiles['q_correct'],
    name='Correct'),
col=1, row=1)

fig.add_trace(go.Scatter(
    y=mean_simulated_quantiles_coherence_tms['q_incorrect'],
    x=observed_quantiles['q_incorrect'],
    name='Incorrect'),
col=2, row=1)

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    name="Optimal simulation",
    mode="lines",
    line=dict(color="Black"),opacity=0.26
))
fig.add_shape(type='line',
                x0=0,
                y0=0,
                x1=observed_quantiles['q_correct'].max(),
                y1=observed_quantiles['q_correct'].max(),
                line=dict(color='black'), opacity=0.26,
                row=1,
                col=1)
fig.add_shape(type='line',
                x0=0,
                y0=0,
                x1=observed_quantiles['q_incorrect'].max(),
                y1=observed_quantiles['q_incorrect'].max(),
                line=dict(color='black'),opacity=0.26,
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
fig.update_xaxes(title_text="Observed RT quantiles", range=[0,8])

# Update yaxis properties
fig.update_yaxes(title_text="Simulated RT quantiles",range=[0,8])

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=size_font, 
    title_font_color="black",  
    title='A. HDDM-4 (Motion coherence - PDI groups)',    
    legend_title_font_color="blue",
    legend_font_size=size_font
)

fig.show()
#%%

data=pd.read_csv('data_rdm_dem.csv')

data=data[data['rt']>0.2]

#Create simulated data for parameter recovery

vat_tms_coh= hddm.load('vat_tms_coh')
vat_tms_coh.print_stats()


trials_per_level = 20
subjs_per_bin=34


tms_s=vat_tms_coh.gen_stats()
tms_s=tms_s.T


#Set up parameters by conditions from previously run model
level_sham_15 = {'v':tms_s.at['mean','v(0.15.0)'],
                 'a':tms_s.at['mean','a(0.15.0)'], 
                 "t":tms_s.at['mean','t(0.15.0)'], 
                 'sv':tms_s.at['std','v(0.15.0)'], 
                 'sa':tms_s.at['std','a(0.15.0)'], 
                 "st":tms_s.at['std','t(0.15.0)'],
                 "p":0.0}

level_sham_05 = {'v':tms_s.at['mean','v(0.05.0)'],
                 'a':tms_s.at['mean','a(0.05.0)'], 
                 "t":tms_s.at['mean','t(0.05.0)'], 
                 'sv':tms_s.at['std','v(0.05.0)'], 
                 'sa':tms_s.at['std','a(0.05.0)'], 
                 "st":tms_s.at['std','t(0.05.0)'],
                                  "p":0.0}


level_tms_15 = {'v':tms_s.at['mean','v(0.15.1)'],
                 'a':tms_s.at['mean','a(0.15.1)'], 
                 "t":tms_s.at['mean','t(0.15.1)'], 
                 'sv':tms_s.at['std','v(0.15.1)'], 
                 'sa':tms_s.at['std','a(0.15.1)'], 
                 "st":tms_s.at['std','t(0.15.1)'],
                                  "p":0.0}


level_tms_05 = {'v':tms_s.at['mean','v(0.05.1)'],
                 'a':tms_s.at['mean','a(0.05.1)'], 
                 "t":tms_s.at['mean','t(0.05.1)'], 
                 'sv':tms_s.at['std','v(0.05.1)'], 
                 'sa':tms_s.at['std','a(0.05.1)'], 
                 "st":tms_s.at['std','t(0.05.1)'],
                                  "p":0.0}

data_sim, params = hddm.generate.gen_rand_data({'level_sham_15':level_sham_15,
                                              'level_sham_05':level_sham_05,
                                         'level_tms_15':level_tms_15,
                                              'level_tms_05':level_tms_05},
                                              size = trials_per_level,
                                              subjs=subjs_per_bin)


data_sim.loc[data_sim['condition'] == 'level_sham_15', 'cond'] = '0.15'
data_sim.loc[data_sim['condition'] == 'level_tms_15', 'cond'] = '0.15'
data_sim.loc[data_sim['condition'] == 'level_sham_05', 'cond'] = '0.05'
data_sim.loc[data_sim['condition'] == 'level_tms_05', 'cond'] = '0.05'

data_sim.loc[data_sim['condition'] == 'level_sham_15', 'session'] = '0'
data_sim.loc[data_sim['condition'] == 'level_sham_05', 'session'] = '0'
data_sim.loc[data_sim['condition'] == 'level_tms_15', 'session'] = '1'
data_sim.loc[data_sim['condition'] == 'level_tms_05', 'session'] = '1'
    
data_sim=data_sim[['rt','response','session','cond']]

data_sim= data_sim[(data_sim['rt'] > 0.2)].assign(rt_type = "simulated")

#plot actual and simulated rt
rt_act=data[['rt','response']]
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

fig.show()
 

#%%
vat_tms_sim = hddm.HDDM(data_sim,depends_on={'v': ['cond','session'], 'a':['cond','session'],'t':['cond','session']},
                        std_depends=True, p_outlier=0.10)
vat_tms_sim.find_starting_values()
vat_tms_sim.sample(20000, burn=2000, thin=5,  dbname='vat_tms_sim.db', db='pickle')
vat_tms_sim.save('vat_tms_sim')
vat_tms_sim.print_stats()
vat_tms_sim.plot_posteriors()
vat_tms_sim.plot_posterior_predictive()


#%%
vat_tms_sim= hddm.load('vat_tms_sim')

vat_tms_sim_traces = vat_tms_sim.get_traces()

vat_tms_coh_traces = vat_tms_coh.get_traces()


#get the dataframe columns
cols = vat_tms_sim_traces.columns 

#print the columns
print(cols)

###preparing dataset for plotting

par_est=vat_tms_coh_traces.assign(Parameter = "estimated")


par_est_15_sham=par_est[['v(0.15.0.0)','a(0.15.0.0)','t(0.15.0.0)','Parameter']].assign(Condition = "High Precision", Session= "Sham")
par_est_15_sham.rename(columns = {'v(0.15.0.0)':'v','a(0.15.0.0)':'a','t(0.15.0.0)':'t',}, inplace = True)


par_est_05_sham=par_est[['v(0.05.0.0)','a(0.05.0.0)','t(0.05.0.0)','Parameter']].assign(Condition = "Low Precision", Session= "Sham")
par_est_05_sham.rename(columns = {'v(0.05.0.0)':'v','a(0.05.0.0)':'a','t(0.05.0.0)':'t',}, inplace = True)


par_est_15_tms=par_est[['v(0.15.1.0)','a(0.15.1.0)','t(0.15.1.0)','Parameter']].assign(Condition = "High Precision", Session= "TMS")
par_est_15_tms.rename(columns = {'v(0.15.1.0)':'v','a(0.15.1.0)':'a','t(0.15.1.0)':'t',}, inplace = True)


par_est_05_tms=par_est[['v(0.05.1.0)','a(0.05.1.0)','t(0.05.1.0)','Parameter']].assign(Condition = "Low Precision", Session= "TMS")
par_est_05_tms.rename(columns = {'v(0.05.1.0)':'v','a(0.05.1.0)':'a','t(0.05.1.0)':'t',}, inplace = True)


par_est=pd.concat([par_est_15_sham, par_est_05_sham,
                  par_est_15_tms,par_est_05_tms])


par_sim=vat_tms_sim_traces.assign(Parameter = "simulated")


par_sim_15_sham=par_sim[['v(0.15.0)','a(0.15.0)','t(0.15.0)','Parameter']].assign(Condition = "High Precision", Session= "Sham")
par_sim_15_sham.rename(columns = {'v(0.15.0)':'v','a(0.15.0)':'a','t(0.15.0)':'t',}, inplace = True)


par_sim_05_sham=par_sim[['v(0.05.0)','a(0.05.0)','t(0.05.0)','Parameter']].assign(Condition = "Low Precision", Session= "Sham")
par_sim_05_sham.rename(columns = {'v(0.05.0)':'v','a(0.05.0)':'a','t(0.05.0)':'t',}, inplace = True)


par_sim_15_tms=par_sim[['v(0.15.1)','a(0.15.1)','t(0.15.1)','Parameter']].assign(Condition = "High Precision", Session= "TMS")
par_sim_15_tms.rename(columns = {'v(0.15.1)':'v','a(0.15.1)':'a','t(0.15.1)':'t',}, inplace = True)


par_sim_05_tms=par_sim[['v(0.05.1)','a(0.05.1)','t(0.05.1)','Parameter']].assign(Condition = "Low Precision", Session= "TMS")
par_sim_05_tms.rename(columns = {'v(0.05.1)':'v','a(0.05.1)':'a','t(0.05.1)':'t',}, inplace = True)



par_sim=pd.concat([par_sim_15_sham, par_sim_05_sham,
                  par_sim_15_tms,par_sim_05_tms])


par_comp=pd.concat([par_est, par_sim])
#%%
def calculate_dev(y_true,y_pred):
    return (2*(y_true * np.log(y_true/y_pred) - (y_true-y_pred))).sum()

#####v
fig = px.histogram(par_comp, x="v", color='Parameter',  
                facet_col="Session",
                facet_row='Condition',
                histnorm='probability density',
                orientation='v',
                opacity=0.4,
                hover_data=par_comp.columns,
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


#####a
fig = px.histogram(par_comp, x="a", color='Parameter',  
                facet_col="Session",
                facet_row='Condition',
                histnorm='probability density',
                orientation='v',
                opacity=0.4,
                hover_data=par_comp.columns,
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



#####t
fig = px.histogram(par_comp, x="t", color='Parameter',  
                facet_col="Session",
                facet_row='Condition',
                histnorm='probability density',
                orientation='v',
                opacity=0.4,
                hover_data=par_comp.columns,
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


############################### DRIFT RATE

v_15_0, v_15_1, v_5_0, v_5_1 = vat_tms_coh.nodes_db.node[['v(0.15.0)', 'v(0.15.1)','v(0.05.0)', 'v(0.05.1)']]
vs_15_0, vs_15_1, vs_5_0, vs_5_1 = vat_tms_sim.nodes_db.node[['v(level_sham_15)', 'v(level_tms_15)',
                                                              'v(level_sham_05)', 'v(level_tms_05)']]


vposteriors= hddm.analyze.plot_posterior_nodes([v_15_0, vs_15_0, v_5_0, vs_5_0], bins=15)
plt.legend(['Estimated','Simulated', 'Estimated','Simulated' ],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for Sham')
print ("P(v High Precision: Sham >  TMS)=",(v_15_0.trace()< vs_15_0.trace()).mean())
print ("P(v Low Precision: Sham >  TMS)=",(v_5_0.trace()< vs_5_0.trace()).mean())


vposteriors= hddm.analyze.plot_posterior_nodes([v_15_1, vs_15_1, v_5_1, vs_5_1], bins=15)
plt.legend(['Estimated','Simulated', 'Estimated','Simulated' ],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for TMS')
print ("P(v High Precision: Sham >  TMS)=",(v_15_1.trace()< vs_15_1.trace()).mean())
print ("P(v Low Precision: Sham >  TMS)=",(v_5_1.trace()< vs_5_1.trace()).mean())



v_sham=[vat_tms_coh_traces[['v(0.15.0)']], vat_tms_sim_traces[['v(0.15.0)']], 
        vat_tms_coh_traces[['v(0.05.0)']], vat_tms_sim_traces[['v(0.05.0)']]]

v_tms=[vat_tms_coh_traces[['v(0.15.1)']], vat_tms_sim_traces[['v(0.15.1)']], 
        vat_tms_coh_traces[['v(0.05.1)']], vat_tms_sim_traces[['v(0.05.1)']]]



v_coh=vat_tms_coh_traces[['v(0.15.0)', 'v(0.15.1)',
                          'v(0.05.0)', 'v(0.05.1)']]

v_s=vat_tms_sim_traces[['v(level_sham_15)', 'v(level_tms_15)','v(level_sham_05)', 'v(level_tms_05)']]


############SHAM
##v
fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=v_sham,  palette='crest',linewidth=0.5, inner='box')
ax.set_xlabel('High Precision                                                 Low Precision', size=32, weight="bold",labelpad=80)
ax.set_ylabel("Drift-rate", size=24, weight="bold",labelpad=30)
ax.set_xticklabels(['Estimated','Simulated','Estimated','Simulated' ],size=24, weight="bold")
plt.title('Sham',size=32, weight="bold")


############TMS
##v
fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=v_tms,  palette='crest',linewidth=0.5, inner='box')
ax.set_xlabel('High Precision                                                 Low Precision', size=32, weight="bold",labelpad=80)
ax.set_ylabel("Drift-rate", size=24, weight="bold",labelpad=30)
ax.set_xticklabels(['Estimated','Simulated','Estimated','Simulated' ],size=24, weight="bold")
plt.title('TMS',size=32, weight="bold")


############################### DECISION THRESHOLD


a_15_0, a_15_1, a_5_0, a_5_1= vat_tms_coh.nodes_db.node[['a(0.15.0)', 'a(0.15.1)','a(0.05.0)', 'a(0.05.1)']]
as_15_0, as_15_1, as_5_0, as_5_1= vat_tms_sim.nodes_db.node[['a(level_sham_15)', 'a(level_tms_15)',
                                                             'a(level_sham_05)', 'a(level_tms_05)']]

aposteriors= hddm.analyze.plot_posterior_nodes([a_15_0, as_15_0, a_5_0, as_5_0], bins=15)
plt.legend(['Estimated','Simulated', 'Estimated','Simulated' ],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Decision threshold')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for Sham')
print ("P(v High Precision: Sham >  TMS)=",(a_15_0.trace()< as_15_0.trace()).mean())
print ("P(v Low Precision: Sham >  TMS)=",(a_5_0.trace()< as_5_0.trace()).mean())


aposteriors= hddm.analyze.plot_posterior_nodes([a_15_1, as_15_1, a_5_1, as_5_1], bins=15)
plt.legend(['Estimated','Simulated', 'Estimated','Simulated' ],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Decision threshold')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for TMS')
print ("P(v High Precision: Sham >  TMS)=",(a_15_1.trace()< as_15_1.trace()).mean())
print ("P(v Low Precision: Sham >  TMS)=",(a_5_1.trace()< as_5_1.trace()).mean())



a_sham=[vat_tms_coh_traces[['a(0.15.0)']], vat_tms_sim_traces[['a(level_sham_15)']], 
        vat_tms_coh_traces[['a(0.05.0)']], vat_tms_sim_traces[['a(level_sham_05)']]]

a_tms=[vat_tms_coh_traces[['a(0.15.1)']], vat_tms_sim_traces[['a(level_tms_15)']], 
        vat_tms_coh_traces[['a(0.05.1)']], vat_tms_sim_traces[['a(level_tms_05)']]]


############SHAM
##a
fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=a_sham,  palette='crest',linewidth=0.5, inner='box')
ax.set_xlabel('High Precision                                                 Low Precision', size=32, weight="bold",labelpad=80)
ax.set_ylabel("Decision threshold", size=24, weight="bold",labelpad=30)
ax.set_xticklabels(['Estimated','Simulated','Estimated','Simulated' ],size=24, weight="bold")
plt.title('Sham',size=32, weight="bold")


############TMS
##a
fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=a_tms, palette='crest', inner='box')
ax.set_xlabel('High Precision                                                 Low Precision', size=32, weight="bold",labelpad=80)
ax.set_ylabel("Decision threshold", size=24, weight="bold",labelpad=30)
ax.set_xticklabels(['Estimated','Simulated','Estimated','Simulated' ],size=24, weight="bold")
plt.title('TMS',size=32, weight="bold")


############################### NON-DECISION TIME 

t_15_0, t_15_1, t_5_0, t_5_1 = vat_tms_coh.nodes_db.node[['t(0.15.0)', 't(0.15.1)','t(0.05.0)', 't(0.05.1)']]
ts_15_0, ts_15_1, ts_5_0, ts_5_1 = vat_tms_sim.nodes_db.node[['t(level_sham_15)', 't(level_tms_15)',
                                                              't(level_sham_05)', 't(level_tms_05)']]


tposteriors= hddm.analyze.plot_posterior_nodes([t_15_0, ts_15_0, t_5_0, ts_5_0], bins=15)
plt.legend(['Estimated','Simulated', 'Estimated','Simulated' ],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Non-decision time')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for Sham')
print ("P(v High Precision: Sham >  TMS)=",(t_15_0.trace()< ts_15_0.trace()).mean())
print ("P(v Low Precision: Sham >  TMS)=",(t_5_0.trace()< ts_5_0.trace()).mean())


tposteriors= hddm.analyze.plot_posterior_nodes([t_15_1, ts_15_1, t_5_1, ts_5_1], bins=15)
plt.legend(['Estimated','Simulated', 'Estimated','Simulated' ],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Non-decision time')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for TMS')
print ("P(v High Precision: Sham >  TMS)=",(t_15_1.trace()< ts_15_1.trace()).mean())
print ("P(v Low Precision: Sham >  TMS)=",(t_5_1.trace()< ts_5_1.trace()).mean())



t_sham=[vat_tms_coh_traces[['t(0.15.0)']], vat_tms_sim_traces[['t(level_sham_15)']], 
        vat_tms_coh_traces[['t(0.05.0)']], vat_tms_sim_traces[['t(level_sham_05)']]]

t_tms=[vat_tms_coh_traces[['t(0.15.1)']], vat_tms_sim_traces[['t(level_tms_15)']], 
        vat_tms_coh_traces[['t(0.05.1)']], vat_tms_sim_traces[['t(level_tms_05)']]]


############SHAM
##t
fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=t_sham,  palette='crest',inner='box')
ax.set_xlabel('High Precision                                                 Low Precision', size=32, weight="bold",labelpad=80)
ax.set_ylabel("Non-decision time", size=24, weight="bold",labelpad=30)
ax.set_xticklabels(['Estimated','Simulated','Estimated','Simulated' ],size=24, weight="bold")
plt.title('Sham',size=32, weight="bold")


############TMS
##t
fig, ax = plt.subplots(figsize=(24, 18))
ax = sns.violinplot( data=t_tms, palette='crest', inner='box')
ax.set_xlabel('High Precision                                                 Low Precision', size=32, weight="bold",labelpad=80)
ax.set_ylabel("Non-decision time", size=24, weight="bold",labelpad=30)
ax.set_xticklabels(['Estimated','Simulated','Estimated','Simulated' ],size=24, weight="bold")
plt.title('TMS',size=32, weight="bold")




