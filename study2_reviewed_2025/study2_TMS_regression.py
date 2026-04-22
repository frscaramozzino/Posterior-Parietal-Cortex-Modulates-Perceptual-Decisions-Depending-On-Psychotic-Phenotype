#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 16:48:49 2023

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
#%% 
#plotting rts with outliers

    
data_rdm_dem=pd.read_csv('data_rdm_dem.csv')

data_rdm_dem['session']=data_rdm_dem['session'].astype('object')
data_rdm_dem['coherence']=data_rdm_dem['coherence'].astype('object')

new_data_rdm_dem = data_rdm_dem[(data_rdm_dem['rt'] > 0.2)] 


new_data_rdm_dem['z_pdi']=(new_data_rdm_dem.pdi - new_data_rdm_dem.pdi.mean())/new_data_rdm_dem.pdi.std(ddof=0)
new_data_rdm_dem['z_caps']=(new_data_rdm_dem.caps - new_data_rdm_dem.caps.mean())/new_data_rdm_dem.caps.std(ddof=0)


#z-scoring the variable




new_data_rdm_dem['z_age'] = (new_data_rdm_dem['age'] - 
                             new_data_rdm_dem['age'].mean()) / new_data_rdm_dem['age'].std()



new_data_rdm_dem['z_pdi']=(new_data_rdm_dem['pdi'] - 
                             new_data_rdm_dem['pdi'].mean()) / new_data_rdm_dem['pdi'].std()


new_data_rdm_dem['z_caps']=(new_data_rdm_dem['caps'] - 
                             new_data_rdm_dem['caps'].mean()) / new_data_rdm_dem['caps'].std()



new_data_rdm_dem.dropna()
#%%
#cond
mreg_avt= hddm.models.HDDMRegressor(new_data_rdm_dem,
                                             ['a ~ 1', 'v ~ 1', 't ~ 1'], group_only_regressors=True,p_outlier=0.10)
mreg_avt.find_starting_values()
mreg_avt.sample(20000, burn=2000,thin=5, dbname='mreg_avt_traces.db', db='pickle')
mreg_avt.save('mreg_avt')

#%%

#cond
mreg_avt_cond= hddm.models.HDDMRegressor(new_data_rdm_dem,
                                             ['a ~ 1+coherence', 'v ~ 1+coherence', 't ~ 1+coherence'], group_only_regressors=True,p_outlier=0.10)
mreg_avt_cond.find_starting_values()
mreg_avt_cond.sample(20000, burn=2000,thin=5, dbname='mreg_avt_cond_traces.db', db='pickle')
mreg_avt_cond.save('mreg_avt_cond')

#%%
mreg_avt_cond = hddm.load('mreg_avt_cond')
mreg_avt_cond.plot_posteriors()

mreg_avt_cond.print_stats()

#%%
#cond,TMS
mreg_avt_tms_cond= hddm.models.HDDMRegressor(new_data_rdm_dem,
                                             ['a ~ 1+coherence+session', 'v ~ 1+coherence+session', 't ~ 1++coherence+session'], group_only_regressors=True,p_outlier=0.10)
mreg_avt_tms_cond.find_starting_values()
mreg_avt_tms_cond.sample(20000, burn=2000,thin=5, dbname='mreg_avt_tms_cond_traces.db', db='pickle')
mreg_avt_tms_cond.save('mreg_avt_tms_cond')

mreg_avt_tms_cond = hddm.load('mreg_avt_tms_cond')
mreg_avt_tms_cond.plot_posteriors()

mreg_avt_tms_cond.print_stats()


a_tms= mreg_avt_tms_cond.nodes_db.node ['a_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([a_tms], bins=8)
plt.legend(['β for TMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (a_tms.trace() > 0).mean())

v_tms= mreg_avt_tms_cond.nodes_db.node ['v_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([v_tms], bins=8)
plt.legend(['β for TMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (v_tms.trace() < 0).mean())

a_cond= mreg_avt_tms_cond.nodes_db.node ['a_coherence[T.0.15]'] 
hddm.analyze.plot_posterior_nodes ([a_cond], bins=8)
plt.legend(['β for Condition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (a_cond.trace() > 0).mean())

v_cond = mreg_avt_tms_cond.nodes_db.node ['v_coherence[T.0.15]'] 
hddm.analyze.plot_posterior_nodes ([v_cond], bins=8)
plt.legend(['β for Condition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (v_cond.trace() < 0).mean())

#%%
#cond,TMS interactuib
mreg_avt_tms_cond2= hddm.models.HDDMRegressor(new_data_rdm_dem, ['a ~ 1+coherence+session+coherence*session', 
                                                                 'v ~ 1++coherence+session+coherence*session',
                                                                  't ~ 1++coherence+session+coherence*session'], group_only_regressors=True,p_outlier=0.10)
mreg_avt_tms_cond2.find_starting_values()
mreg_avt_tms_cond2.sample(20000, burn=2000,thin=5, dbname='mreg_avt_tms_cond2_traces.db', db='pickle')
mreg_avt_tms_cond2.save('mreg_avt_tms_cond2')

mreg_avt_tms_cond2 = hddm.load('mreg_avt_tms_cond2')
mreg_avt_tms_cond2.plot_posteriors()

mreg_avt_tms_cond2.print_stats()

a_tms= mreg_avt_tms_cond2.nodes_db.node ['a_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([a_tms], bins=8)
plt.legend(['β for TMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (a_tms.trace() > 0).mean())

v_tms= mreg_avt_tms_cond2.nodes_db.node ['v_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([v_tms], bins=8)
plt.legend(['β for TMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (v_tms.trace() < 0).mean())

t_tms= mreg_avt_tms_cond2.nodes_db.node ['t_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([t_tms], bins=8)
plt.legend(['β for TMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (t_tms.trace() < 0).mean())

a_cond= mreg_avt_tms_cond2.nodes_db.node ['a_coherence[T.0.15]'] 
hddm.analyze.plot_posterior_nodes ([a_cond], bins=8)
plt.legend(['β for Condition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (a_cond.trace() > 0).mean())

v_cond = mreg_avt_tms_cond2.nodes_db.node ['v_coherence[T.0.15]'] 
hddm.analyze.plot_posterior_nodes ([v_cond], bins=8)
plt.legend(['β for Condition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (v_cond.trace() < 0).mean())


t_cond = mreg_avt_tms_cond2.nodes_db.node ['t_coherence[T.0.15]'] 
hddm.analyze.plot_posterior_nodes ([t_cond], bins=8)
plt.legend(['β for Condition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (t_cond.trace() < 0).mean())

a_condxsession= mreg_avt_tms_cond2.nodes_db.node ['a_coherence[T.0.15]:session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([a_condxsession], bins=8)
plt.legend(['β for TMSxCondition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (a_condxsession.trace() > 0).mean())

v_condxsession = mreg_avt_tms_cond2.nodes_db.node ['v_coherence[T.0.15]:session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([v_condxsession], bins=8)
plt.legend(['β for TMSxCondition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (v_condxsession.trace() < 0).mean())

t_condxsession = mreg_avt_tms_cond2.nodes_db.node ['t_coherence[T.0.15]:session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([v_condxsession], bins=8)
plt.legend(['β for TMSxCondition'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (t_condxsession.trace() < 0).mean())


new_data_rdm_dem=new_data_rdm_dem.dropna()
#%%
new_data_rdm_dem=new_data_rdm_dem.dropna()

#cond,PDI
mreg_avt_coh_tms_PDI= hddm.models.HDDMRegressor(new_data_rdm_dem, ['a ~ 1+coherence+z_pdi+session+z_pdi*session', 
                                                               'v ~ 1+coherence+z_pdi+session+z_pdi*session', 
                                                               't ~ 1+coherence+z_pdi+session+z_pdi*session'], group_only_regressors=True,p_outlier=0.10)
mreg_avt_coh_tms_PDI.find_starting_values()
mreg_avt_coh_tms_PDI.sample(20000, burn=2000,thin=5, dbname='mreg_avt_coh_tms_PDI_traces.db', db='pickle')
mreg_avt_coh_tms_PDI.save('mreg_avt_coh_tms_PDI')

mreg_avt_coh_tms_PDI = hddm.load('mreg_avt_coh_tms_PDI')

mreg_avt_coh_tms_PDI.print_stats()
#%%
new_data_rdm_dem=new_data_rdm_dem.dropna()

#cond,PDI
mreg_avt_coh_tms_PDI= hddm.models.HDDMRegressor(new_data_rdm_dem, ['a ~ 1+coherence+z_pdi+session+z_pdi*session', 
                                                               'v ~ 1+coherence+z_pdi+session+z_pdi*session', 
                                                               't ~ 1+coherence+z_pdi+session+z_pdi*session'], group_only_regressors=True,p_outlier=0.10)
mreg_avt_coh_tms_PDI.find_starting_values()
mreg_avt_coh_tms_PDI.sample(20000, burn=2000,thin=5, dbname='mreg_avt_coh_tms_PDI_traces.db', db='pickle')
mreg_avt_coh_tms_PDI.save('mreg_avt_coh_tms_PDI')

mreg_avt_coh_tms_PDI = hddm.load('mreg_avt_coh_tms_PDI')

mreg_avt_coh_tms_PDI.print_stats()
#%%
a_PDI= mreg_avt_coh_tms_PDI.nodes_db.node ['a_z_pdi'] 
hddm.analyze.plot_posterior_nodes ([a_PDI], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (a_PDI.trace() > 0).mean())

v_PDI = mreg_avt_coh_tms_PDI.nodes_db.node ['v_z_pdi'] 
hddm.analyze.plot_posterior_nodes ([v_PDI], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (v_PDI.trace() < 0).mean())

t_PDI = mreg_avt_coh_tms_PDI.nodes_db.node ['t_z_pdi'] 
hddm.analyze.plot_posterior_nodes ([t_PDI], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (t_PDI.trace() < 0).mean())
#%%


a_tms= mreg_avt_coh_tms_PDI.nodes_db.node ['a_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([a_tms], bins=8)
plt.legend(['β for TMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (a_tms.trace() > 0).mean())

v_tms= mreg_avt_coh_tms_PDI.nodes_db.node ['v_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([v_tms], bins=8)
plt.legend(['β for TMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI > 0) = ", (v_tms.trace() >0).mean())

t_tms= mreg_avt_coh_tms_PDI.nodes_db.node ['t_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([t_tms], bins=8)
plt.legend(['β for TMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (t_tms.trace() < 0).mean())

#%%

a_PDI_tms= mreg_avt_coh_tms_PDI.nodes_db.node ['a_z_pdi:session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([a_PDI_tms], bins=8)
plt.legend(['β for PDIxTMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (a_PDI_tms.trace() > 0).mean())

v_PDI_tms = mreg_avt_coh_tms_PDI.nodes_db.node ['v_z_pdi:session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([v_PDI_tms], bins=8)
plt.legend(['β for PDIxTMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (v_PDI_tms.trace() < 0).mean())

t_PDI_tms = mreg_avt_coh_tms_PDI.nodes_db.node ['t_z_pdi:session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([t_PDI_tms], bins=8)
plt.legend(['β for PDIxTMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (t_PDI_tms.trace() > 0).mean())

#%%
#cond,PDI

mreg_avt_tms_PDI= hddm.models.HDDMRegressor(new_data_rdm_dem, ['a ~ 1+z_pdi+session+z_pdi*session', 
                                                               'v ~ 1+z_pdi+session+z_pdi*session', 
                                                               't ~ 1+z_pdi+session+z_pdi*session'], group_only_regressors=True,p_outlier=0.10)
mreg_avt_tms_PDI.find_starting_values()
mreg_avt_tms_PDI.sample(20000, burn=2000,thin=5, dbname='mreg_avt_tms_PDI_traces.db', db='pickle')
mreg_avt_tms_PDI.save('mreg_avt_tms_PDI')

mreg_avt_tms_PDI = hddm.load('mreg_avt_tms_PDI')
mreg_avt_tms_PDI.plot_posteriors()

mreg_avt_tms_PDI.print_stats()


#%%
a_PDI= mreg_avt_tms_PDI.nodes_db.node ['a_z_pdi'] 
hddm.analyze.plot_posterior_nodes ([a_PDI], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (a_PDI.trace() > 0).mean())

v_PDI = mreg_avt_tms_PDI.nodes_db.node ['v_z_pdi'] 
hddm.analyze.plot_posterior_nodes ([v_PDI], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (v_PDI.trace() < 0).mean())

t_PDI = mreg_avt_tms_PDI.nodes_db.node ['t_z_pdi'] 
hddm.analyze.plot_posterior_nodes ([t_PDI], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (t_PDI.trace() < 0).mean())
#%%


a_tms= mreg_avt_tms_PDI.nodes_db.node ['a_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([a_tms], bins=8)
plt.legend(['β for TMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (a_tms.trace() > 0).mean())

v_tms= mreg_avt_tms_PDI.nodes_db.node ['v_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([v_tms], bins=8)
plt.legend(['β for TMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI > 0) = ", (v_tms.trace() >0).mean())

t_tms= mreg_avt_tms_PDI.nodes_db.node ['t_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([t_tms], bins=8)
plt.legend(['β for TMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (t_tms.trace() < 0).mean())



a_PDI_tms= mreg_avt_tms_PDI.nodes_db.node ['a_z_pdi:session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([a_PDI_tms], bins=8)
plt.legend(['β for PDIxTMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_PDI > 0) = ", (a_PDI_tms.trace() > 0).mean())

v_PDI_tms = mreg_avt_tms_PDI.nodes_db.node ['v_z_pdi:session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([v_PDI_tms], bins=8)
plt.legend(['β for PDIxTMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (v_PDI_tms.trace() < 0).mean())

t_PDI_tms = mreg_avt_tms_PDI.nodes_db.node ['t_z_pdi:session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([t_PDI_tms], bins=8)
plt.legend(['β for PDIxTMS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_PDI < 0) = ", (t_PDI_tms.trace() > 0).mean())


#%%
#################
#cond,CAPS
mreg_avt_tms_CAPS= hddm.models.HDDMRegressor(new_data_rdm_dem, ['a ~ 1+z_caps+session+z_caps*session', 
                                                                'v ~ 1++z_caps+session+z_caps*session', 
                                                                't ~ 1++z_caps+session+z_caps*session'], group_only_regressors=True,p_outlier=0.10)
mreg_avt_tms_CAPS.find_starting_values()
mreg_avt_tms_CAPS.sample(20000, burn=2000,thin=5, dbname='mreg_avt_tms_CAPS_traces.db', db='pickle')
mreg_avt_tms_CAPS.save('mreg_avt_tms_CAPS')
 
mreg_avt_tms_CAPS = hddm.load('mreg_avt_tms_CAPS')
#%%
mreg_avt_tms_CAPS.print_stats()


a_CAPS= mreg_avt_tms_CAPS.nodes_db.node ['a_z_caps'] 
hddm.analyze.plot_posterior_nodes ([a_CAPS], bins=8)
plt.legend([u'β for CAPS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_CAPS > 0) = ", (a_CAPS.trace() > 0).mean())
print ("P(a_CAPS > 0) = ", (a_CAPS.trace() < 0).mean())

v_CAPS = mreg_avt_tms_CAPS.nodes_db.node ['v_z_caps'] 
hddm.analyze.plot_posterior_nodes ([v_CAPS], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_CAPS < 0) = ", (v_CAPS.trace() < 0).mean())
print ("P(v_CAPS < 0) = ", (v_CAPS.trace() > 0).mean())

t_CAPS = mreg_avt_tms_CAPS.nodes_db.node ['t_z_caps'] 
hddm.analyze.plot_posterior_nodes ([t_CAPS], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(t_CAPS < 0) = ", (t_CAPS.trace() < 0).mean())
print ("P(t_CAPS < 0) = ", (t_CAPS.trace() > 0).mean())


a_tms= mreg_avt_tms_CAPS.nodes_db.node ['a_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([a_tms], bins=8)
plt.legend([u'β for CAPS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_CAPS > 0) = ", (a_tms.trace() > 0).mean())
print ("P(a_CAPS > 0) = ", (a_tms.trace() < 0).mean())

v_tms = mreg_avt_tms_CAPS.nodes_db.node ['v_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([v_tms], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_CAPS < 0) = ", (v_tms.trace() < 0).mean())
print ("P(v_CAPS < 0) = ", (v_tms.trace() > 0).mean())

t_tms = mreg_avt_tms_CAPS.nodes_db.node ['t_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([t_tms], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(t_CAPS < 0) = ", (t_tms.trace() < 0).mean())
print ("P(t_CAPS < 0) = ", (t_tms.trace() > 0).mean())


a_CAPS_tms= mreg_avt_tms_CAPS.nodes_db.node ['a_z_caps:session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([a_CAPS_tms], bins=8)
plt.legend([u'β for CAPS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_CAPS > 0) = ", (a_CAPS_tms.trace() > 0).mean())
print ("P(a_CAPS > 0) = ", (a_CAPS_tms.trace() < 0).mean())

v_CAPS_tms = mreg_avt_tms_CAPS.nodes_db.node ['v_z_caps:session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([v_CAPS], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_CAPS < 0) = ", (v_CAPS_tms.trace() < 0).mean())
print ("P(v_CAPS < 0) = ", (v_CAPS_tms.trace() > 0).mean())

t_CAPS_tms = mreg_avt_tms_CAPS.nodes_db.node ['t_z_caps:session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([t_CAPS_tms], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_CAPS < 0) = ", (t_CAPS_tms.trace() < 0).mean())
print ("P(v_CAPS < 0) = ", (t_CAPS_tms.trace() > 0).mean())


#%%
#cond,CAPS
mreg_avt_coh_tms_CAPS= hddm.models.HDDMRegressor(new_data_rdm_dem, ['a ~ 1+coherence+z_caps+session+z_caps*session', 
                                                               'v ~ 1+coherence+z_caps+session+z_caps*session', 
                                                               't ~ 1+coherence+z_caps+session+z_caps*session'], group_only_regressors=True,p_outlier=0.10)
mreg_avt_coh_tms_CAPS.find_starting_values()
mreg_avt_coh_tms_CAPS.sample(20000, burn=2000,thin=5, dbname='mreg_avt_coh_tms_CAPS_traces.db', db='pickle')
mreg_avt_coh_tms_CAPS.save('mreg_avt_coh_tms_CAPS')

mreg_avt_coh_tms_CAPS = hddm.load('mreg_avt_coh_tms_CAPS')
#%%
mreg_avt_coh_tms_CAPS.print_stats()


a_CAPS= mreg_avt_coh_tms_CAPS.nodes_db.node ['a_z_caps'] 
hddm.analyze.plot_posterior_nodes ([a_CAPS], bins=8)
plt.legend([u'β for CAPS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_CAPS > 0) = ", (a_CAPS.trace() > 0).mean())
print ("P(a_CAPS > 0) = ", (a_CAPS.trace() < 0).mean())

v_CAPS = mreg_avt_coh_tms_CAPS.nodes_db.node ['v_z_caps'] 
hddm.analyze.plot_posterior_nodes ([v_CAPS], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_CAPS < 0) = ", (v_CAPS.trace() < 0).mean())
print ("P(v_CAPS < 0) = ", (v_CAPS.trace() > 0).mean())

t_CAPS = mreg_avt_coh_tms_CAPS.nodes_db.node ['t_z_caps'] 
hddm.analyze.plot_posterior_nodes ([t_CAPS], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(t_CAPS < 0) = ", (t_CAPS.trace() < 0).mean())
print ("P(t_CAPS < 0) = ", (t_CAPS.trace() > 0).mean())

#%%
a_tms= mreg_avt_coh_tms_CAPS.nodes_db.node ['a_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([a_tms], bins=8)
plt.legend([u'β for CAPS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_CAPS > 0) = ", (a_tms.trace() > 0).mean())
print ("P(a_CAPS > 0) = ", (a_tms.trace() < 0).mean())

v_tms = mreg_avt_coh_tms_CAPS.nodes_db.node ['v_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([v_tms], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_CAPS < 0) = ", (v_tms.trace() < 0).mean())
print ("P(v_CAPS < 0) = ", (v_tms.trace() > 0).mean())

t_tms = mreg_avt_coh_tms_CAPS.nodes_db.node ['t_session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([t_tms], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(t_CAPS < 0) = ", (t_tms.trace() < 0).mean())
print ("P(t_CAPS < 0) = ", (t_tms.trace() > 0).mean())

#%%
a_CAPS_tms= mreg_avt_coh_tms_CAPS.nodes_db.node ['a_z_caps:session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([a_CAPS_tms], bins=8)
plt.legend([u'β for CAPS'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1, fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(a_CAPS > 0) = ", (a_CAPS_tms.trace() > 0).mean())
print ("P(a_CAPS > 0) = ", (a_CAPS_tms.trace() < 0).mean())

v_CAPS_tms = mreg_avt_coh_tms_CAPS.nodes_db.node ['v_z_caps:session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([v_CAPS], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_CAPS < 0) = ", (v_CAPS_tms.trace() < 0).mean())
print ("P(v_CAPS < 0) = ", (v_CAPS_tms.trace() > 0).mean())

t_CAPS_tms = mreg_avt_coh_tms_CAPS.nodes_db.node ['t_z_caps:session[T.1.0]'] 
hddm.analyze.plot_posterior_nodes ([t_CAPS_tms], bins=8)
plt.legend(['β for PDI'],loc=0, bbox_to_anchor= (1.01, 1.01), ncol=1,fontsize="large", borderaxespad=0, frameon=False)
plt.xlabel('β value')
plt.ylabel('Posterior Probability Density ')
print ("P(v_CAPS < 0) = ", (t_CAPS_tms.trace() < 0).mean())
print ("P(v_CAPS < 0) = ", (t_CAPS_tms.trace() > 0).mean())




