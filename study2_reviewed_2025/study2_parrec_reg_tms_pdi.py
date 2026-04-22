#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 17:07:23 2023

@author: francescoscaramozzino
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 14:49:00 2023

@author: francescoscaramozzino
"""


import hddm
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from scipy.stats import zscore

from tableone import TableOne, load_dataset
from scipy.special import rel_entr
import random
import numpy as np
########################################################
  
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

#plotting rts with outliers

    
data_rdm_dem=pd.read_csv('data_rdm_dem.csv')

data_rdm_dem['session']=data_rdm_dem['session'].astype('object')
data_rdm_dem['coherence']=data_rdm_dem['coherence'].astype('object')

new_data_rdm_dem = data_rdm_dem[(data_rdm_dem['rt'] > 0.2)] 


mreg_avt_tms_PDI = hddm.load('mreg_avt_tms_PDI')
mreg_avt_tms_PDI.print_stats()

statsy=mreg_avt_tms_PDI.gen_stats()
stats_s=statsy.T


"""
intercept and betas for drift rate
"""
int_v=stats_s.at['mean','v_Intercept']
 
#TMS
b_v_tms=stats_s.at['mean','v_session[T.1.0]']

#trail condition varying within subjects
b_v_cond=stats_s.at['mean','v_z_pdi']


#trail condition varying within subjects
b_v_condxtms=stats_s.at['mean','v_z_pdi:session[T.1.0]']

"""
intercept and betas for decision threshold
"""
int_a=stats_s.at['mean','a_Intercept']
int_a_sd=stats_s.at['std','a_Intercept']

#CAPS varying between subjects
b_a_tms=stats_s.at['mean','a_session[T.1.0]']

#trail condition varying within subjects
b_a_cond=stats_s.at['mean','a_z_pdi']


#trail condition varying within subjects
b_a_condxtms=stats_s.at['mean','a_z_pdi:session[T.1.0]']


"""
intercept and betas for non-decision time
"""
int_t=stats_s.at['mean','a_Intercept']
int_t_sd=stats_s.at['std','a_Intercept']

#CAPS varying between subjects
b_t_tms=stats_s.at['mean','t_session[T.1.0]']

#trail condition varying within subjects
b_t_cond=stats_s.at['mean','t_z_pdi']


#trail condition varying within subjects
b_t_condxtms=stats_s.at['mean','t_z_pdi:session[T.1.0]']


#Generate data

trials_per_level = 40
subjs_per_bin=68



#values of parameters as intercept+betas for two levels: 0, condition beta=0, so only CAPS beta; 1, condition + CAPS beta. 

x=new_data_rdm_dem['pdi'].mean()

a_05_0= int_a
v_05_0= int_v
t_05_0= int_t

a_05_1= int_a+b_a_tms
v_05_1= int_v+b_v_tms
t_05_1= int_t+b_t_tms

a_15_0= int_a+b_a_cond
v_15_0= int_v+b_v_cond
t_15_0= int_t+b_t_cond

a_15x1= int_a+b_a_cond+b_a_tms+b_a_condxtms
v_15x1= int_v+b_v_cond+b_v_tms+b_v_condxtms
t_15x1= int_t+b_t_cond+b_t_tms+b_t_condxtms


mreg_avt_tms_PDI_traces=mreg_avt_tms_PDI.get_traces()



    
parvec_0,parvec_01, parvec_02,parvec_03,parvec_04,parvec_05,parvec_06,parvec_07,parvec_08,parvec_09

parvec_05_0 = {'v':v_05_0, 'a':a_05_0, 't':t_05_0,'sv':0, 
                 'sa':0, 
                 "st":0} 
parvec_05_1 = {'v':v_05_1, 'a':a_05_1, 't':t_05_1,'sv':0, 
                 'sa':0, 
                 "st":0}  
parvec_15_0 = {'v':v_15_0, 'a':a_15_0, 't':t_15_0,'sv':0, 
                 'sa':0, 
                 "st":0} 

parvec_15x1 = {'v':v_15x1, 'a':a_15x1, 't':t_15x1,'sv':0, 
                 'sa':0, 
                 "st":0} 




data_sim, params_sim = hddm.generate.gen_rand_data({'level_15x1': parvec_15x1},
                                                   size=trials_per_level,
                                                   subjs=subjs_per_bin)


data_sim = data_sim.assign(tms = '0')
data_sim.loc[data_sim['condition'] == 'level_05_1', 'tms'] = '1'
data_sim.loc[data_sim['condition'] == 'level_15_1', 'tms'] = '1'
data_sim.loc[data_sim['condition'] == 'level_15x1', 'tms'] = '1'

#simulating CPAS values by randomly sampling from our data

df_pdi=new_data_rdm_dem.pdi.dropna()
f=pd.Series(df_pdi.sample(n=68,replace=True)).reset_index().dropna()
pdi_sim=pd.concat([f]*40)#to have a value of CAPS for each trial
pdi_sim=pdi_sim.loc[:,'pdi']

data_sim['x_pdi']= zscore(pdi_sim)
data_sim=data_sim[['rt','response','tms','x_pdi']]


data_sim = data_sim[(data_sim['rt'] > 0.2)]
 


#plot simuleted against original data
rt_act=new_data_rdm_dem[['rt','response']]
rt_act=rt_act.assign(rt_type = "actual")

rt_sim=data_sim[['rt']]
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

#cond,CAPS recovery model 

sim_mreg_avt_tms_pdi= hddm.models.HDDMRegressor(data_sim,
                                             ['a ~ 1+x_pdi+tms+x_pdi*tms', 
                                              'v ~ 1+x_pdi+tms+x_pdi*tms', 
                                              't ~ 1+x_pdi+tms+x_pdi*tms'], 
                                             group_only_regressors=True,p_outlier=0.10)
sim_mreg_avt_tms_pdi.find_starting_values()
sim_mreg_avt_tms_pdi.sample(20000, burn=2000,thin=5, 
                              dbname='sim_mreg_avt_tms_pdi_traces.db', db='pickle')
sim_mreg_avt_tms_pdi.save('sim_mreg_avt_tms_pdi')

sim_mreg_avt_tms_pdi = hddm.load('sim_mreg_avt_tms_pdi')
sim_mreg_avt_tms_pdi.plot_posteriors()

sim_mreg_avt_tms_pdi.print_stats()



"""
Evalutating differences between actual and synthetic parameters 
"""
sim_a_TMS= sim_mreg_avt_tms_cond2.nodes_db.node ['a_tms[T.1]'] 
hddm.analyze.plot_posterior_nodes ([sim_a_TMS], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(B > 0) = ", (sim_a_TMS.trace() > 0).mean())

sim_v_TMS = sim_mreg_avt_tms_cond2.nodes_db.node ['v_tms[T.1]'] 
hddm.analyze.plot_posterior_nodes ([sim_v_TMS], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(B < 0) = ", (sim_v_TMS.trace() < 0).mean())

sim_t_TMS = sim_mreg_avt_tms_cond2.nodes_db.node ['t_tms[T.1]'] 
hddm.analyze.plot_posterior_nodes ([sim_t_TMS], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(B < 0) = ", (sim_t_TMS.trace() < 0).mean())

sim_a_COND= sim_mreg_avt_tms_cond2.nodes_db.node ['a_cond[T.15]'] 
hddm.analyze.plot_posterior_nodes ([sim_a_COND], bins=8)
plt.legend(['β for Condition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (sim_a_COND.trace() > 0).mean())

sim_v_COND = sim_mreg_avt_tms_cond2.nodes_db.node ['v_cond[T.15]'] 
hddm.analyze.plot_posterior_nodes ([sim_v_COND], bins=8)
plt.legend(['β for Condition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (sim_v_COND.trace() < 0).mean())

sim_t_COND = sim_mreg_avt_tms_cond2.nodes_db.node ['t_cond[T.15]'] 
hddm.analyze.plot_posterior_nodes ([sim_t_COND], bins=8)
plt.legend(['β for Condition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (sim_t_COND.trace() < 0).mean())

Sim_a_CONDxTMS= sim_mreg_avt_tms_cond2.nodes_db.node ['a_cond[T.15]:tms[T.1]'] 
hddm.analyze.plot_posterior_nodes ([sim_a_COND], bins=8)
plt.legend(['β for Condition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (sim_a_COND.trace() > 0).mean())

sim_v_CONDxTMS = sim_mreg_avt_tms_cond2.nodes_db.node ['v_cond[T.15]:tms[T.1]'] 
hddm.analyze.plot_posterior_nodes ([sim_v_COND], bins=8)
plt.legend(['β for Condition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (sim_v_COND.trace() < 0).mean())

sim_t_CONDxTMD = sim_mreg_avt_tms_cond2.nodes_db.node ['t_cond[T.15]:tms[T.1]'] 
hddm.analyze.plot_posterior_nodes ([sim_t_COND], bins=8)
plt.legend(['β for Condition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (sim_t_COND.trace() < 0).mean())


a_TMS= mreg_avt_tms_cond2.nodes_db.node ['a_session[T.1.0]'] 
v_TMS = mreg_avt_tms_cond2.nodes_db.node ['v_session[T.1.0]'] 
t_TMS = mreg_avt_tms_cond2.nodes_db.node ['t_session[T.1.0]'] 

a_COND =mreg_avt_tms_cond2.nodes_db.node ['a_coherence[T.0.15]'] 
v_COND = mreg_avt_tms_cond2.nodes_db.node ['v_coherence[T.0.15]'] 
t_COND = mreg_avt_tms_cond2.nodes_db.node ['t_coherence[T.0.15]'] 

a_CONDxTMS =mreg_avt_tms_cond2.nodes_db.node ['a_coherence[T.0.15]:session[T.1.0]'] 
v_CONDxTMS = mreg_avt_tms_cond2.nodes_db.node ['v_coherence[T.0.15]:session[T.1.0]'] 
t_CONDxTMS = mreg_avt_tms_cond2.nodes_db.node ['v_coherence[T.0.15]:session[T.1.0]'] 

#plot betas
mreg_avt_tms_cond_traces=mreg_avt_tms_cond2.get_traces()

a_tms=pd.DataFrame(mreg_avt_tms_cond_traces['a_session[T.1.0]']).assign(β="TMS", model = "estimated",y= "Decision threshold")
a_tms.rename(columns = {'a_session[T.1.0]':'beta'}, inplace = True)

v_tms=pd.DataFrame(mreg_avt_tms_cond_traces['v_session[T.1.0]']).assign(β="TMS",model = "estimated",y= "Drift rate")
v_tms.rename(columns = {'v_session[T.1.0]':'beta'}, inplace = True)

t_tms=pd.DataFrame(mreg_avt_tms_cond_traces['t_session[T.1.0]']).assign(β="TMS",model = "estimated",y= "Non-decision time")
t_tms.rename(columns = {'t_session[T.1.0]':'beta'}, inplace = True)

a_cond=pd.DataFrame(mreg_avt_tms_cond_traces['a_coherence[T.0.15]']).assign(β="Condition",model = "estimated",y= "Decision threshold")
a_cond.rename(columns = {'a_coherence[T.0.15]':'beta'}, inplace = True)

v_cond=pd.DataFrame(mreg_avt_tms_cond_traces['v_coherence[T.0.15]']).assign(β="Condition",model = "estimated",y= "Drift rate")
v_cond.rename(columns = {'v_coherence[T.0.15]':'beta'}, inplace = True)

t_cond=pd.DataFrame(mreg_avt_tms_cond_traces['t_coherence[T.0.15]']).assign(β="Condition",model = "estimated",y= "Non-decision time")
t_cond.rename(columns = {'t_coherence[T.0.15]':'beta'}, inplace = True)

a_condxtms=pd.DataFrame(mreg_avt_tms_cond_traces['a_coherence[T.0.15]:session[T.1.0]']).assign(β="TMSxCondition",model = "estimated",y= "Decision threshold")
a_condxtms.rename(columns = {'a_coherence[T.0.15]:session[T.1.0]':'beta'}, inplace = True)

v_condxtms=pd.DataFrame(mreg_avt_tms_cond_traces['v_coherence[T.0.15]:session[T.1.0]']).assign(β="TMSxCondition",model = "estimated",y= "Drift rate")
v_condxtms.rename(columns = {'v_coherence[T.0.15]:session[T.1.0]':'beta'}, inplace = True)

t_condxtms=pd.DataFrame(mreg_avt_tms_cond_traces['t_coherence[T.0.15]:session[T.1.0]']).assign(β="TMSxCondition",model = "estimated",y= "Non-decision time")
t_condxtms.rename(columns = {'t_coherence[T.0.15]:session[T.1.0]':'beta'}, inplace = True)

sim_mreg_avt_tms_cond_traces=sim_mreg_avt_tms_cond2.get_traces()

sim_a_tms=pd.DataFrame(sim_mreg_avt_tms_cond_traces['a_tms[T.1]']).assign(β="TMS", model = "simulated",y= "Decision threshold")
sim_a_tms.rename(columns = {'a_tms[T.1]':'beta'}, inplace = True)

sim_v_tms=pd.DataFrame(sim_mreg_avt_tms_cond_traces['v_tms[T.1]']).assign(β="TMS", model = "simulated",y= "Drift rate")
sim_v_tms.rename(columns = {'v_tms[T.1]':'beta'}, inplace = True)

sim_t_tms=pd.DataFrame(sim_mreg_avt_tms_cond_traces['t_tms[T.1]']).assign(β="TMS", model = "simulated",y= "Non-decision time")
sim_t_tms.rename(columns = {'t_tms[T.1]':'beta'}, inplace = True)

sim_a_cond=pd.DataFrame(sim_mreg_avt_tms_cond_traces['a_cond[T.15]']).assign(β="Condition", model = "simulated",y= "Decision threshold")
sim_a_cond.rename(columns = {'a_cond[T.15]':'beta'}, inplace = True)

sim_v_cond=pd.DataFrame(sim_mreg_avt_tms_cond_traces['v_cond[T.15]']).assign(β="Condition", model = "simulated",y= "Drift rate")
sim_v_cond.rename(columns = {'v_cond[T.15]':'beta'}, inplace = True)

sim_t_cond=pd.DataFrame(sim_mreg_avt_tms_cond_traces['t_cond[T.15]']).assign(β="Condition", model = "simulated",y= "Non-decision time")
sim_t_cond.rename(columns = {'t_cond[T.15]':'beta'}, inplace = True)

sim_a_condxtms=pd.DataFrame(sim_mreg_avt_tms_cond_traces['a_cond[T.15]:tms[T.1]']).assign(β="TMSxCondition", model = "simulated",y= "Decision threshold")
sim_a_condxtms.rename(columns = {'a_cond[T.15]:tms[T.1]':'beta'}, inplace = True)

sim_v_condxtms=pd.DataFrame(sim_mreg_avt_tms_cond_traces['v_cond[T.15]:tms[T.1]']).assign(β="TMSxCondition", model = "simulated",y= "Drift rate")
sim_v_condxtms.rename(columns = {'v_cond[T.15]:tms[T.1]':'beta'}, inplace = True)

sim_t_condxtms=pd.DataFrame(sim_mreg_avt_tms_cond_traces['t_cond[T.15]:tms[T.1]']).assign(β="TMSxCondition", model = "simulated",y= "Non-decision time")
sim_t_condxtms.rename(columns = {'t_cond[T.15]:tms[T.1]':'beta'}, inplace = True)


#plot betas for PDI

betas=pd.concat([v_cond, v_tms,v_condxtms,
                 a_cond, a_tms,a_condxtms,
                 t_cond, t_tms,t_condxtms,
                 sim_v_cond, sim_v_tms,sim_v_condxtms,
                sim_a_cond, sim_a_tms,sim_a_condxtms,
                 sim_t_cond, sim_t_tms,sim_t_condxtms])

#######prob density plots
coloursdist={
         "Condition": "blue",
         "TMS": "red",
         "TMSxCondition": "fuchsia"}
#####

fig = px.histogram(betas, x="beta", color='model',  
                facet_col="y", facet_row="β",
                facet_row_spacing=0.08,
                histnorm='probability density',
                orientation='v',
                barmode='group',opacity=0.5,
                labels= {'beta':'β value'},
                width=1700,
                height=800)
fig.update_yaxes(matches=None)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=22,
    legend_title_font_color="blue",
    legend_font_size=22
)
fig.update_xaxes(title_font_family="Arial")

fig.add_vline(x=0., line_width=1)
