#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 19:23:12 2023

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
import numpy as np
########################################################
  
#%%
data=pd.read_csv('data_rdm_dem.csv')

data=data[data['rt']>0.2]
# data_err = data[data['response'] == 0]
# data_cor = data[data['response'] == 1]

# out_err=data_err['rt'].quantile(.9)
# out_cor=data_cor['rt'].quantile(.9)
# data = data[data['rt'] <= out_err]

#%%

vat_tms_coh = hddm.HDDM(data,depends_on={'v': ['coherence','session'], 
                                                'a':['coherence','session'],
                                                't':['coherence','session']}, std_depends=True, p_outlier=0.10)
vat_tms_coh.find_starting_values()
vat_tms_coh.sample(20000, burn=2000, thin=5,  dbname='vat_tms_coh_traces.db', db='pickle')
vat_tms_coh.save('vat_tms_coh')

vat_tms_coh.print_stats()

vat_tms_coh2 = hddm.HDDM(data,depends_on={'v': ['coherence','session'], 
                                                'a':['coherence','session'],
                                                }, std_depends=True, p_outlier=0.10)
vat_tms_coh2.find_starting_values()
vat_tms_coh2.sample(20000, burn=2000, thin=5,  dbname='vat_tms_coh2_traces.db', db='pickle')
vat_tms_coh2.save('vat_tms_coh2')
vat_tms_coh =vat_tms_coh2@
vat_tms_coh2.print_stats()
#%%

#model TMS vs Shamx Low PDI vs High PDI

vat_tms_pdi = hddm.HDDM(data,depends_on={'v': ['pdi_group','session'], 
                                                'a':['pdi_group','session'],
                                                't':['pdi_group','session']}, std_depends=True, p_outlier=0.10)
vat_tms_pdi.find_starting_values()
vat_tms_pdi.sample(20000, burn=2000, thin=5,  dbname='vat_tms_pdi_traces.db', db='pickle')
vat_tms_pdi.save('vat_tms_pdi')
vat_tms_pdi.print_stats()


vat_tms_pdi2 = hddm.HDDM(data,depends_on={'v': ['pdi_group','session'], 
                                                'a':['pdi_group','session']}, std_depends=True, p_outlier=0.10)
vat_tms_pdi2.find_starting_values()
vat_tms_pdi2.sample(20000, burn=2000, thin=5,  dbname='vat_tms_pdi2_traces.db', db='pickle')
vat_tms_pdi2.save('vat_tms_pdi2')
vat_tms_pdi2.print_stats()

#%%

#model TMS vs Shamx Low CAPS vs High CAPS

vat_tms_caps = hddm.HDDM(data,depends_on={'v': ['caps_group','session'], 
                                                'a':['caps_group','session'],
                                                't':['caps_group','session']}, std_depends=True, p_outlier=0.10)
vat_tms_caps.find_starting_values()
vat_tms_caps.sample(20000, burn=2000, thin=5,  dbname='vat_tms_caps_traces.db', db='pickle')
vat_tms_caps.save('vat_tms_caps')

vat_tms_caps2 = hddm.HDDM(data,depends_on={'v': ['caps_group','session'], 
                                                'a':['caps_group','session']
                                               }, std_depends=True, p_outlier=0.10)
vat_tms_caps2.find_starting_values()
vat_tms_caps2.sample(20000, burn=2000, thin=5,  dbname='vat_tms_caps2_traces.db', db='pickle')
vat_tms_caps2.save('vat_tms_caps2')


#%%

#Results tms coherence

vat_tms_coh= hddm.load('vat_tms_coh')
vat_tms_coh.print_stats()


v_15_0, v_15_1, v_5_0, v_5_1 = vat_tms_coh.nodes_db.node[['v(0.15.0.0)', 'v(0.15.1.0)','v(0.05.0.0)', 'v(0.05.1.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([v_15_0, v_15_1, v_5_0, v_5_1], bins=15)
plt.legend(['v Sham x High Precision','v TMS x High Precision', 'v Sham x Low Precision','v TMS x Low Precision'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for SessionxCondition')
print ("P(v High Precision: Sham >  TMS)=",(v_15_0.trace()> v_15_1.trace()).mean())
print ("P(v Low Precision: Sham >  TMS)=",(v_5_0.trace()> v_5_1.trace()).mean())

a_15_0, a_15_1, a_5_0, a_5_1= vat_tms_coh.nodes_db.node[['a(0.15.0.0)', 'a(0.15.1.0)','a(0.05.0.0)', 'a(0.05.1.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([a_15_0, a_15_1, a_5_0, a_5_1], bins=15)
plt.legend(['a Sham x High Precision','a TMS x High Precision', 'a Sham x Low Precision','a TMS x Low Precision'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for SessionxCondition')
print ("P(a High Precision: Sham > TMS)=",(a_15_0.trace()> a_15_1.trace()).mean())
print ("P(a Low Precision: Sham >  TMS)=",(a_5_0.trace()> a_5_1.trace()).mean())


t_15_0, t_15_1, t_5_0, t_5_1 = vat_tms_coh.nodes_db.node[['t(0.15.0.0)', 't(0.15.1.0)','t(0.05.0.0)', 't(0.05.1.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([t_15_0, t_15_1, t_5_0, t_5_1 ], bins=15)
plt.legend(['t Sham x High Precision','t TMS x High Precision', 't Sham x Low Precision','t TMS x Low Precision'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior decision-threshold distributions for SessionxCondition')
print ("P(t High Precision: Sham > TMS)=",(t_15_0.trace()> t_15_1.trace()).mean())
print ("P(t Low Precision: Sham > TMS)=",(t_5_0.trace()> t_5_1.trace()).mean())


#%%

#Results tms PDI
vat_tms_pdi= hddm.load('vat_tms_pdi')
vat_tms_pdi.print_stats()

v_LPDI_0, v_HPDI_1, v_LPDI_1, v_HPDI_0 = vat_tms_pdi.nodes_db.node[['v(low_pdi.0.0)', 'v(high_pdi.1.0)','v(low_pdi.1.0)', 'v(high_pdi.0.0)']]

PDI_vposteriors= hddm.analyze.plot_posterior_nodes([v_LPDI_0, v_LPDI_1, v_HPDI_0, v_HPDI_1], bins=15)
plt.legend(['v ShamxLPDI','v TMSxLPDI', 'v ShamxHPDI','v TMSxHPDI'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition')
print ("P(v LPDI: Sham > v TMS)=",(v_LPDI_0.trace()< v_LPDI_1.trace()).mean())
print ("P(v HPDI: Sham > v TMS)=",(v_HPDI_0.trace()< v_HPDI_1.trace()).mean())
print ("P(v Sham: LPDI > v HPDI)=",(v_LPDI_0.trace()< v_HPDI_0.trace()).mean())
print ("P(v TMS: LPDI > v HPDI)=",(v_LPDI_1.trace()< v_HPDI_1.trace()).mean())


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

#Results tms CAPS
vat_tms_caps= hddm.load('vat_tms_caps')
vat_tms_caps.print_stats()

v_LCAPS_0, v_HCAPS_1, v_LCAPS_1, v_HCAPS_0 = vat_tms_caps.nodes_db.node[['v(low_caps.0.0)', 'v(high_caps.1.0)','v(low_caps.1.0)', 'v(high_caps.0.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([v_LCAPS_0, v_LCAPS_1, v_HCAPS_0, v_HCAPS_1], bins=15)
plt.legend(['v ShamxLCAPS','v TMSxLCAPS', 'v ShamxHCAPS','v TMSxHCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition')
print ("P(v LCAPS: Sham > v TMS)=",(v_LCAPS_0.trace()< v_LCAPS_1.trace()).mean())
print ("P(v HCAPS: Sham > v TMS)=",(v_HCAPS_0.trace()< v_HCAPS_1.trace()).mean())
print ("P(v Sham: LCAPS > v HCAPS)=",(v_LCAPS_0.trace()< v_HCAPS_0.trace()).mean())
print ("P(v TMS: LCAPS > v HCAPS)=",(v_LCAPS_1.trace()< v_HCAPS_1.trace()).mean())


a_LCAPS_0, a_HCAPS_1, a_LCAPS_1, a_HCAPS_0 = vat_tms_caps.nodes_db.node[['a(low_caps.0.0)', 'a(high_caps.1.0)','a(low_caps.1.0)', 'a(high_caps.0.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([a_LCAPS_0, a_LCAPS_1, a_HCAPS_0, a_HCAPS_1], bins=15)
plt.legend(['a ShamxLCAPS','a TMSxLCAPS', 'a ShamxHCAPS','a TMSxHCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Decision-threshold')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition')
print ("P(a LCAPS: Sham > a TMS)=",(a_LCAPS_0.trace()> a_LCAPS_1.trace()).mean())
print ("P(a HCAPS: Sham > a TMS)=",(a_HCAPS_0.trace()> a_HCAPS_1.trace()).mean())


t_LCAPS_0, t_HCAPS_1, t_LCAPS_1, t_HCAPS_0 = vat_tms_caps.nodes_db.node[['t(low_caps.0.0)', 't(high_caps.1.0)','t(low_caps.1.0)', 't(high_caps.0.0)']]
PDI_vposteriors= hddm.analyze.plot_posterior_nodes([t_LCAPS_0, t_LCAPS_1, t_HCAPS_0, t_HCAPS_1], bins=15)
plt.legend(['t ShamxLCAPS','t TMSxLCAPS', 't ShamxHCAPS','t TMSxHCAPS'],loc=2, bbox_to_anchor= (1.01, 1.01), ncol=1, borderaxespad=0, frameon=False)
plt.xlabel('Non-decision time')
plt.ylabel('Posterior probability')
plt.title('Posterior drift-rate distributions for PDIxCondition')
print ("P(t LCAPS: Sham > t TMS)=",(t_LCAPS_0.trace()> t_LCAPS_1.trace()).mean())
print ("P(t HCAPS: Sham > t TMS)=",(t_HCAPS_0.trace()> t_HCAPS_1.trace()).mean())



