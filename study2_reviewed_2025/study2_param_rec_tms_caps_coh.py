#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 16:26:54 2023

@author: francescoscaramozzino
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 15:33:56 2022

@author: francescoscaramozzino
"""


import hddm
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import scipy.stats as stats
from tableone import TableOne, load_dataset
from scipy.special import rel_entr
import random

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

trials_per_level = 20
subjs_per_bin=34

data=pd.read_csv('data_rdm_dem.csv')

data=data[data['rt']>0.2]
# data_err = data[data['response'] == 0]
# data_cor = data[data['response'] == 1]

# out_err=data_err['rt'].quantile(.9)
# out_cor=data_cor['rt'].quantile(.9)
# data = data[data['rt'] <= out_err]


mvat_tms_caps= hddm.load('vat_tms_caps_coh') 
caps_s=mvat_tms_caps.gen_stats()
caps_s=caps_s.T
mvat_tms_caps.print_stats()

#%%
#Set up parameters by conditions from previously run model
level_sham_high_caps_lc = {'v':caps_s.at['mean','v(high_caps.0.05.0.0)'],
                 'a':caps_s.at['mean','a(high_caps.0.05.0.0)'], 
                 "t":caps_s.at['mean','t(high_caps.0.05.0.0)'], 
                 'sv':0,
                 'sz':0, 
                 'sa':0, 
                 "st":0,
                 "p":0.1}

level_sham_low_caps_lc= {'v':caps_s.at['mean','v(low_caps.0.05.0.0)'],
                 'a':caps_s.at['mean','a(low_caps.0.05.0.0)'], 
                 "t":caps_s.at['mean','t(low_caps.0.05.0.0)'], 
                 'sv':0,
                 'sz':0, 
                 'sa':0, 
                 "st":0,
                 "p":0.1}

level_sham_high_caps_hc = {'v':caps_s.at['mean','v(high_caps.0.15.0.0)'],
                 'a':caps_s.at['mean','a(high_caps.0.15.0.0)'], 
                 "t":caps_s.at['mean','t(high_caps.0.15.0.0)'], 
                 'sv':0,
                 'sz':0, 
                 'sa':0, 
                 "st":0,
                 "p":0.1}

level_sham_low_caps_hc= {'v':caps_s.at['mean','v(low_caps.0.15.0.0)'],
                 'a':caps_s.at['mean','a(low_caps.0.15.0.0)'], 
                 "t":caps_s.at['mean','t(low_caps.0.15.0.0)'], 
                 'sv':0,
                 'sz':0, 
                 'sa':0, 
                 "st":0,
                 "p":0.1}

level_tms_high_caps_lc = {'v':caps_s.at['mean','v(high_caps.0.05.1.0)'],
                 'a':caps_s.at['mean','a(high_caps.0.05.1.0)'], 
                 "t":caps_s.at['mean','t(high_caps.0.05.1.0)'], 
                 'sv':0,
                 'sz':0, 
                 'sa':0, 
                 "st":0,
                 "p":0.1}

level_tms_low_caps_lc = {'v':caps_s.at['mean','v(low_caps.0.05.1.0)'],
                 'a':caps_s.at['mean','a(low_caps.0.05.1.0)'], 
                 "t":caps_s.at['mean','t(low_caps.0.05.1.0)'], 
                 'sv':0,
                 'sz':0, 
                 'sa':0, 
                 "st":0,
                 "p":0.1}

level_tms_high_caps_hc = {'v':caps_s.at['mean','v(high_caps.0.15.1.0)'],
                 'a':caps_s.at['mean','a(high_caps.0.15.1.0)'], 
                 "t":caps_s.at['mean','t(high_caps.0.15.1.0)'], 
                 'sv':0,
                 'sz':0, 
                 'sa':0, 
                 "st":0,
                 "p":0.1}

level_tms_low_caps_hc = {'v':caps_s.at['mean','v(low_caps.0.15.1.0)'],
                 'a':caps_s.at['mean','a(low_caps.0.15.1.0)'], 
                 "t":caps_s.at['mean','t(low_caps.0.15.1.0)'], 
                 'sv':0,
                 'sz':0, 
                 'sa':0, 
                 "st":0,
                 "p":0.1}


#%%
  



data_sim, params = hddm.generate.gen_rand_data({'level_sham_high_caps_lc':level_sham_high_caps_lc,
                                                  'level_sham_low_caps_lc':level_sham_low_caps_lc,
                                              'level_tms_high_caps_lc':level_tms_high_caps_lc,
                                                  'level_tms_low_caps_lc':level_tms_low_caps_lc,
                                                  'level_sham_high_caps_hc':level_sham_high_caps_hc,
                                                  'level_sham_low_caps_hc':level_sham_low_caps_hc,
                                              'level_tms_high_caps_hc':level_tms_high_caps_hc,
                                                  'level_tms_low_caps_hc':level_tms_low_caps_hc},
                                                  size = trials_per_level,
                                                  subjs=subjs_per_bin)
    
data_sim.loc[data_sim['condition'] == 'level_sham_high_caps_lc', 'caps_group'] = 'high_caps'
data_sim.loc[data_sim['condition'] == 'level_sham_low_caps_lc', 'caps_group'] = 'low_caps'
data_sim.loc[data_sim['condition'] == 'level_tms_high_caps_lc', 'caps_group'] = 'high_caps'
data_sim.loc[data_sim['condition'] == 'level_tms_low_caps_lc', 'caps_group'] = 'low_caps'
data_sim.loc[data_sim['condition'] == 'level_sham_high_caps_hc', 'caps_group'] = 'high_caps'
data_sim.loc[data_sim['condition'] == 'level_sham_low_caps_hc', 'caps_group'] = 'low_caps'
data_sim.loc[data_sim['condition'] == 'level_tms_high_caps_hc', 'caps_group'] = 'high_caps'
data_sim.loc[data_sim['condition'] == 'level_tms_low_caps_hc', 'caps_group'] = 'low_caps'

data_sim.loc[data_sim['condition'] == 'level_sham_high_caps_lc', 'session'] = '0'
data_sim.loc[data_sim['condition'] == 'level_sham_low_caps_lc', 'session'] = '0'
data_sim.loc[data_sim['condition'] == 'level_tms_high_caps_lc', 'session'] = '1'
data_sim.loc[data_sim['condition'] == 'level_tms_low_caps_lc', 'session'] = '1'

data_sim.loc[data_sim['condition'] == 'level_sham_high_caps_hc', 'session'] = '0'
data_sim.loc[data_sim['condition'] == 'level_sham_low_caps_hc', 'session'] = '0'
data_sim.loc[data_sim['condition'] == 'level_tms_high_caps_hc', 'session'] = '1'
data_sim.loc[data_sim['condition'] == 'level_tms_low_caps_hc', 'session'] = '1'

data_sim.loc[data_sim['condition'] == 'level_sham_high_caps_lc', 'coherence'] = '0.05'
data_sim.loc[data_sim['condition'] == 'level_sham_low_caps_lc', 'coherence'] = '0.05'
data_sim.loc[data_sim['condition'] == 'level_tms_high_caps_lc', 'coherence'] = '0.05'
data_sim.loc[data_sim['condition'] == 'level_tms_low_caps_lc', 'coherence'] = '0.05'

data_sim.loc[data_sim['condition'] == 'level_sham_high_caps_hc', 'coherence'] = '0.15'
data_sim.loc[data_sim['condition'] == 'level_sham_low_caps_hc', 'coherence'] = '0.15'
data_sim.loc[data_sim['condition'] == 'level_tms_high_caps_hc', 'coherence'] = '0.15'
data_sim.loc[data_sim['condition'] == 'level_tms_low_caps_hc', 'coherence'] = '0.15'

data_sim=data_sim[['rt','response','session','coherence','caps_group']]

data_sim_caps= data_sim[(data_sim['rt'] > 0.2)].assign(rt_type = "simulated")


#%%

vat_tms_caps_sim = hddm.HDDM(data_sim,depends_on={'v': ['caps_group','session', 'coherence'], 'a':['caps_group','session', 'coherence'],'t':['caps_group','session', 'coherence']},
                        std_depends=True, p_outlier=0.10)
vat_tms_caps_sim.find_starting_values()
vat_tms_caps_sim.sample(20000, burn=2000, thin=5,  dbname='vat_tms_caps__coh_sim_trace.db', db='pickle')
vat_tms_caps_sim.save('vat_tms_caps__coh_sim')

vat_tms_caps_sim= hddm.load('vat_tms_caps__coh_sim')

vat_tms_caps_sim.print_stats()

vat_tms_caps_sim_traces = vat_tms_caps_sim.get_traces()
mvat_tms_caps_traces = mvat_tms_caps.get_traces()

#%%
vat_tms_caps_sim= hddm.load('vat_tms_caps__coh_sim')

par_est=mvat_tms_caps_traces.assign(Parameter = "estimated")


par_est_lp_sham=par_est[['v(low_caps.0.05.0.0)','a(low_caps.0.05.0.0)','t(low_caps.0.05.0.0)','Parameter']].assign(Group = "Low CAPS", Session= "Sham")
par_est_lp_sham.rename(columns = {'v(low_caps.0.05.0.0)':'v','a(low_caps.0.05.0.0)':'a','t(low_caps.0.05.0.0)':'t',}, inplace = True)


par_est_hp_sham=par_est[['v(high_caps.0.05.0.0)','a(high_caps.0.05.0.0)','t(high_caps.0.05.0.0)','Parameter']].assign(Group = "High CAPS", Session= "Sham")
par_est_hp_sham.rename(columns = {'v(high_caps.0.05.0.0)':'v','a(high_caps.0.05.0.0)':'a','t(high_caps.0.05.0.0)':'t',}, inplace = True)


par_est_lp_tms=par_est[['v(low_caps.0.05.1.0)','a(low_caps.0.05.1.0)','t(low_caps.0.05.1.0)','Parameter']].assign(Group = "Low CAPS", Session= "TMS")
par_est_lp_tms.rename(columns = {'v(low_caps.0.05.1.0)':'v','a(low_caps.0.05.1.0)':'a','t(low_caps.0.05.1.0)':'t'}, inplace = True)


par_est_hp_tms=par_est[['v(high_caps.0.05.1.0)','a(high_caps.0.05.1.0)','t(high_caps.0.05.1.0)','Parameter']].assign(Group = "High CAPS", Session= "TMS")
par_est_hp_tms.rename(columns = {'v(high_caps.0.05.1.0)':'v','a(high_caps.0.05.1.0)':'a','t(high_caps.0.05.1.0)':'t'}, inplace = True)


par_est_lc=pd.concat([par_est_lp_sham, par_est_hp_sham,
                  par_est_lp_tms,par_est_hp_tms])

par_est_lp_sham=par_est[['v(low_caps.0.15.0.0)','a(low_caps.0.15.0.0)','t(low_caps.0.15.0.0)','Parameter']].assign(Group = "Low CAPS", Session= "Sham")
par_est_lp_sham.rename(columns = {'v(low_caps.0.15.0.0)':'v','a(low_caps.0.15.0.0)':'a','t(low_caps.0.15.0.0)':'t',}, inplace = True)


par_est_hp_sham=par_est[['v(high_caps.0.15.0.0)','a(high_caps.0.15.0.0)','t(high_caps.0.15.0.0)','Parameter']].assign(Group = "High CAPS", Session= "Sham")
par_est_hp_sham.rename(columns = {'v(high_caps.0.15.0.0)':'v','a(high_caps.0.15.0.0)':'a','t(high_caps.0.15.0.0)':'t',}, inplace = True)


par_est_lp_tms=par_est[['v(low_caps.0.15.1.0)','a(low_caps.0.15.1.0)','t(low_caps.0.15.1.0)','Parameter']].assign(Group = "Low CAPS", Session= "TMS")
par_est_lp_tms.rename(columns = {'v(low_caps.0.15.1.0)':'v','a(low_caps.0.15.1.0)':'a','t(low_caps.0.15.1.0)':'t'}, inplace = True)


par_est_hp_tms=par_est[['v(high_caps.0.15.1.0)','a(high_caps.0.15.1.0)','t(high_caps.0.15.1.0)','Parameter']].assign(Group = "High CAPS", Session= "TMS")
par_est_hp_tms.rename(columns = {'v(high_caps.0.15.1.0)':'v','a(high_caps.0.15.1.0)':'a','t(high_caps.0.15.1.0)':'t'}, inplace = True)


par_est_hc=pd.concat([par_est_lp_sham, par_est_hp_sham,
                  par_est_lp_tms,par_est_hp_tms])


par_est=pd.concat([par_est_hc, par_est_lc])
#%%

par_sim=vat_tms_caps_sim_traces.assign(Parameter = "simulated")


par_sim_lp_sham=par_sim[['v(low_caps.0.05.0)','a(low_caps.0.05.0)','t(low_caps.0.05.0)','Parameter']].assign(Group = "Low CAPS", Session= "Sham")
par_sim_lp_sham.rename(columns = {'v(low_caps.0.05.0)':'v','a(low_caps.0.05.0)':'a','t(low_caps.0.05.0)':'t',}, inplace = True)


par_sim_hp_sham=par_sim[['v(high_caps.0.05.0)','a(high_caps.0.05.0)','t(high_caps.0.05.0)','Parameter']].assign(Group = "High CAPS", Session= "Sham")
par_sim_hp_sham.rename(columns = {'v(high_caps.0.05.0)':'v','a(high_caps.0.05.0)':'a','t(high_caps.0.05.0)':'t',}, inplace = True)


par_sim_lp_tms=par_sim[['v(low_caps.0.05.1)','a(low_caps.0.05.1)','t(low_caps.0.05.1)','Parameter']].assign(Group = "Low CAPS", Session= "TMS")
par_sim_lp_tms.rename(columns = {'v(low_caps.0.05.1)':'v','a(low_caps.0.05.1)':'a','t(low_caps.0.05.1)':'t'}, inplace = True)


par_sim_hp_tms=par_sim[['v(high_caps.0.05.1)','a(high_caps.0.05.1)','t(high_caps.0.05.1)','Parameter']].assign(Group = "High CAPS", Session= "TMS")
par_sim_hp_tms.rename(columns = {'v(high_caps.0.05.1)':'v','a(high_caps.0.05.1)':'a','t(high_caps.0.05.1)':'t'}, inplace = True)


par_sim_hc=pd.concat([par_sim_lp_sham, par_sim_hp_sham,
                  par_sim_lp_tms,par_sim_hp_tms])

par_sim_lp_sham=par_sim[['v(low_caps.0.15.0)','a(low_caps.0.15.0)','t(low_caps.0.15.0)','Parameter']].assign(Group = "Low CAPS", Session= "Sham")
par_sim_lp_sham.rename(columns = {'v(low_caps.0.15.0)':'v','a(low_caps.0.15.0)':'a','t(low_caps.0.15.0)':'t',}, inplace = True)


par_sim_hp_sham=par_sim[['v(high_caps.0.15.0)','a(high_caps.0.15.0)','t(high_caps.0.15.0)','Parameter']].assign(Group = "High CAPS", Session= "Sham")
par_sim_hp_sham.rename(columns = {'v(high_caps.0.15.0)':'v','a(high_caps.0.15.0)':'a','t(high_caps.0.15.0)':'t',}, inplace = True)


par_sim_lp_tms=par_sim[['v(low_caps.0.15.1)','a(low_caps.0.15.1)','t(low_caps.0.15.1)','Parameter']].assign(Group = "Low CAPS", Session= "TMS")
par_sim_lp_tms.rename(columns = {'v(low_caps.0.15.1)':'v','a(low_caps.0.15.1)':'a','t(low_caps.0.15.1)':'t'}, inplace = True)


par_sim_hp_tms=par_sim[['v(high_caps.0.15.1)','a(high_caps.0.15.1)','t(high_caps.0.15.1)','Parameter']].assign(Group = "High CAPS", Session= "TMS")
par_sim_hp_tms.rename(columns = {'v(high_caps.0.15.1)':'v','a(high_caps.0.15.1)':'a','t(high_caps.0.15.1)':'t'}, inplace = True)


par_sim_hc=pd.concat([par_sim_lp_sham, par_sim_hp_sham,
                  par_sim_lp_tms,par_sim_hp_tms])

par_sim=pd.concat([par_sim_hc, par_sim_hc])

#%%



par_comp=pd.concat([par_est, par_sim])