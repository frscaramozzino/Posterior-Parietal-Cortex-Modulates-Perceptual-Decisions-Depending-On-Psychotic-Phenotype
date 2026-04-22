#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 22:05:53 2023

@author: francescoscaramozzino
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 21:30:09 2023

@author: francescoscaramozzino
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 22:40:41 2023

@author: francescoscaramozzino
"""


import hddm
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.stats as stats
from scipy.stats import norm
import statistics
import seaborn as sns

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
########
#Figures

#Get traces of parameters for plotting


vat_tms=hddm.load('vat_tms')

vat_tms.print_stats()
vat_tms_traces = vat_tms.get_traces()

#%%%
###preparing dataset for plotting


par_est_sham_lp=vat_tms_traces[['a(0)','v(0)','t(0)']].assign(Session='Sham')
par_est_sham_lp.rename(columns = {'a(0)':'a','v(0)':'v','t(0)':'t'}, inplace = True)



par_est_sham_hp=vat_tms_traces[['a(0)','v(0)','t(0)']].assign(Session='Sham')
par_est_sham_hp.rename(columns = {'a(0)':'a','v(0)':'v','t(0)':'t'}, inplace = True)


par_est_tms_lp=vat_tms_traces[['a(1)','v(1)','t(1)']].assign(Session='TMS')
par_est_tms_lp.rename(columns = {'a(1)':'a','v(1)':'v','t(1)':'t'}, inplace = True)

par_est_tms_hp=vat_tms_traces[['a(1)','v(1)','t(1)']].assign(Session='TMS')
par_est_tms_hp.rename(columns = {'a(1)':'a','v(1)':'v','t(1)':'t'}, inplace = True)



par_comp=pd.concat([par_est_sham_lp, par_est_sham_hp,
                  par_est_tms_lp,par_est_tms_hp])


# par_est_a=par_est[['a','Session','Condition']].assign(Parameter='Decision threshold')
# par_est_a.rename(columns = {'a':'Estimate'}, inplace = True)

# par_est_v=par_est[['v','Session','Condition']].assign(Parameter='Drift rate')
# par_est_v.rename(columns = {'v':'Estimate'}, inplace = True)

# par_est_t=par_est[['t','Session','Condition']].assign(Parameter='Non-decision time')
# par_est_t.rename(columns = {'t':'Estimate'}, inplace = True)
#par_comp=pd.concat([par_est_a, par_est_v,par_est_t])

#%%
colours={"Sham": "slateblue",
         "TMS": "magenta"
         }
fig = px.violin(par_comp, y="v",  box=True, color='Session',  
                violinmode='overlay',
                hover_data=par_comp.columns,
                color_discrete_map=colours, 
                labels= {'v':'Drift rate'},
                width=1100,
                height=1500)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

v_E, v_S = vat_tms.nodes_db.node[['v(1)', 'v(0)']]
hddm.analyze.plot_posterior_nodes([v_E, v_S], bins=20)
plt.legend(['v TMS','v Sham'])
plt.xlabel('Drift-rate')
plt.ylabel('Posterior probability')
print ("v:P(TMS >Sham)=",(v_E.trace()> v_S.trace()).mean())

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    legendgrouptitle_text="Annotation",
    name="P<0.05",
    mode="markers",
    marker=dict(color="Black", symbol='star', size=15)
))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=35,
    title_font_family="Times New Roman",
    title="A.",
    title_x=0.05,  # Adjust the horizontal position of the title (0 to 1)
    title_y=0.88,  # Adjust the vertical position of the title (0 to 1)
    title_font_color="blue",    legend_title_font_color="blue",
    legend_font_size=35,
    legend=dict(
        orientation="h",    
        entrywidth=0,
        yanchor="bottom",
        y=1.1,
        xanchor="right",
        x=1
    )
)
fig.update_xaxes(title_font_family="Arial")
fig.show()
fig.write_image("study2_fig_v_tms.png",scale=3)

#%%
colours={"Sham": "slateblue",
         "TMS": "magenta"
         }
fig = px.violin(par_comp, y="a",  box=True, color='Session',  
                violinmode='overlay',
                hover_data=par_comp.columns,
                color_discrete_map=colours, 
                labels= {'a':'Decision threshold'},
                width=1100,
                height=1300)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

a_E, a_S = vat_tms.nodes_db.node[['a(1)', 'a(0)']]
hddm.analyze.plot_posterior_nodes([a_E, a_S], bins=20)
plt.legend(['a TMS','a Sham'])
plt.xlabel('Decision-threshold')
plt.ylabel('Posterior probability')
print ("P(TMS < Sham)=",(a_E.trace()< a_S.trace()).mean())

# fig.add_trace(go.Scatter(
#     x=0.5,
#     xref='Paper',
#     y=4.8,
#     legendgroup="significant",
#     legendgrouptitle_text="Annotation",
#     name="P<0.05",
#     mode="markers",
#     marker=dict(color="Black", symbol='star', size=15)
# ))
fig.add_annotation(
    x=0.5
    , y=4.8
    , text='<b>*<b>'
    ,showarrow=False
    , font=dict(size=40, color="black", family="Courier New, monospace")
    , xref='paper')

fig.update_traces(showlegend=False)


fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=35,
    title_font_family="Times New Roman",
    title="B.",
    # title_x=0.05,  # Adjust the horizontal position of the title (0 to 1)
    # title_y=0.85,  # Adjust the vertical position of the title (0 to 1)
    title_font_color="blue",    legend_title_font_color="blue",
    legend_font_size=35,
    legend=dict(
        orientation="h",    
        entrywidth=0,
        yanchor="bottom",
        y=1.05,
        xanchor="right",
        x=1
    )
)
fig.update_xaxes(title_font_family="Arial")
fig.show()
fig.write_image("study2_fig_a_tms.png",scale=3)

#%%
colours={"Sham": "slateblue",
         "TMS": "magenta"
         }
fig = px.violin(par_comp, y="t",  box=True, color='Session',  
                violinmode='overlay',
                hover_data=par_comp.columns,
                color_discrete_map=colours, 
                labels= {'t':'Non-decision time'},
                width=1100,
                height=1300)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

t_E, t_S = vat_tms.nodes_db.node[['t(1)', 't(0)']]
hddm.analyze.plot_posterior_nodes([t_E, t_S], bins=20)
plt.legend(['t TMS','t Sham'])
plt.xlabel('Non-decision Time')
plt.ylabel('Posterior probability')
print ("P(TMS < Sham)=",(t_E.trace()< t_S.trace()).mean())


fig.update_traces(showlegend=False)

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=35,
    title_font_family="Times New Roman",
    title="C.",
    # title_x=0.05,  # Adjust the horizontal position of the title (0 to 1)
    # title_y=0.85,  # Adjust the vertical position of the title (0 to 1)
    title_font_color="blue",    legend_title_font_color="blue",
    legend_font_size=35,
    legend=dict(
        orientation="h",    
        entrywidth=0,
        yanchor="bottom",
        y=1.05,
        xanchor="right",
        x=1
    )
)
fig.update_xaxes(title_font_family="Arial")
fig.show()
fig.write_image("study2_fig_t_tms.png",scale=3)

#%%

# Violin plots TMS vs Condition
# the subplot as shown in the above image


set_font_size=35
#V
# Initialize figure with subplots
fig = make_subplots(
    rows=1, cols=1,
    row_heights=[1000]
)


colours={"Sham": "light blue",
         "TMS": "magenta"
         }


# V Add traces
fig.add_trace(go.Violin(y=par_comp['v'][ par_comp['Session']],
                        legendgroup='pdi', scalegroup='Sham', name='Sham',
                        legendgrouptitle_text="Session", 
                        ), col=1, row=1)

# fig.add_trace(go.Violin(x=par_comp['Session'][ par_comp['Session'] == 'TMS' ],
#                         y=par_comp['v'][ par_comp['Session'] == 'TMS' ],
#                         legendgroup='pdi', scalegroup='TMS', name='TMS',
#                         line_color=colours['TMS']), col=1, row=1)

fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(violinmode='overlay')


# Add interaction line
fig.add_shape(type='line',
               x0='Low Coherence',
               y0=vat_tms_traces['v(1)'].mean(),
               x1='High Coherence',
               y1=vat_tms_traces['v(1)'].mean(),
               line=dict(color=colours['TMS']))

# Add interaction line
fig.add_shape(type='line',
               x0='Low Coherence',
               y0=vat_tms_traces['v(0)'].mean(),
               x1='High Coherence',
               y1=vat_tms_traces['v(0)'].mean(),
               line=dict(color='blue'))
             

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
        legendgrouptitle_text="Annotation",
    legendgroup="significant",
    name="P<0.05",
    mode="lines",
    line=dict(color="Black")
))

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    name="P>0.05",
    mode="lines",
    line=dict(color="Black", dash='dash')
))

fig.add_trace(go.Scatter(
    x=[None],
    y=[None],
    legendgroup="significant",
    legendgrouptitle_text="Annotation",
    name="P<0.05",
    mode="markers",
    marker=dict(color="Black", symbol='star', size=15)
))

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# Update xaxis properties
# Update xaxis properties
fig.update_xaxes(title_text="Motion coherence", col=1, row=1)

# Update yaxis properties
fig.update_yaxes(title_text="Drift rate",range=[0.9, 1.4], col=1, row=1)


# Update title and height
fig.update_xaxes(title_font_family="Arial")

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=set_font_size,
    title_font_family="Times New Roman",
    title_font_color="black",
    legend_title_font_color="blue",
    legend_font_size=set_font_size,
    legend=dict(
        orientation="h",    
        entrywidth=200,
        yanchor="bottom",
        y=1.05,
        xanchor="right",
        x=1
    ), width=1200,  # Set your desired width in pixels
    height=1300  # Set your desired height in pixels
)
fig.update_xaxes(title_font_family="Arial")
fig.show()

# Save fig with DPI=300
fig.write_image("study2_fig_v.png",scale=3)

v_lcoh_0, v_lcoh_1, v_hcoh_0, v_hcoh_1 = vat_tms_traces.nodes_db.node[[
    'v(0)', 'v(1)', 'v(0)', 'v(1)']]

print("P(v Low Coherence: Sham > TMS)=", (v_lcoh_0.trace() > v_lcoh_1.trace()).mean())
print("P(v High Coherence: Sham < TMS)=", (v_hcoh_0.trace() < v_hcoh_1.trace()).mean())

print("P(v Sham: Low > High)=", (v_lcoh_0.trace() > v_hcoh_0.trace()).mean())
print("P(v TMS: Low > High)=", (v_lcoh_1.trace() > v_hcoh_1.trace()).mean())
#%%
set_font_size=35
#A
# Initialize figure with subplots
fig = make_subplots(
    rows=1, cols=1,
    row_heights=[1000]
)


colours={"Sham": "light blue",
         "TMS": "magenta"
         }


# V Add traces
fig.add_trace(go.Violin(x=par_comp['Condition'][ par_comp['Session'] == 'Sham' ],
                        y=par_comp['a'][ par_comp['Session'] == 'Sham' ],
                        legendgroup='pdi', scalegroup='Sham', name='Sham',
                        legendgrouptitle_text="Session", showlegend=False,
                        line_color=colours['Sham']), col=1, row=1)


fig.add_trace(go.Violin(x=par_comp['Condition'][ par_comp['Session'] == 'TMS' ],
                        y=par_comp['a'][ par_comp['Session'] == 'TMS' ],
                        legendgroup='pdi', scalegroup='TMS', name='TMS',showlegend=False,
                        line_color=colours['TMS']), col=1, row=1)

fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(violinmode='overlay')


# Add interaction line
fig.add_shape(type='line',
               x0='Low Coherence',
               y0=vat_tms_coh_traces['a(1)'].mean(),
               x1='High Coherence',
               y1=vat_tms_coh_traces['a(1)'].mean(),
               line=dict(color='magenta'))

# Add interaction line
fig.add_shape(type='line',
               x0='Low Coherence',
               y0=vat_tms_coh_traces['a(0)'].mean(),
               x1='High Coherence',
               y1=vat_tms_coh_traces['a(0)'].mean(),
               line=dict(color='blue',dash='dash'))
                

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# Update xaxis properties
# Update xaxis properties
fig.update_xaxes(title_text="Motion coherence", col=1, row=1)

# Update yaxis properties
fig.update_yaxes(title_text="Deision threshold",range=[2.4, 3.8], col=1, row=1)


# Update title and height
fig.update_xaxes(title_font_family="Arial")

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=set_font_size,
    title_font_family="Times New Roman",
    title_font_color="black",
    legend_title_font_color="blue",
    legend_font_size=set_font_size,
    legend=dict(
        orientation="h",    
        entrywidth=400,
        yanchor="bottom",
        y=0.8,
        xanchor="right",
        x=2.5
    ), width=1200,  # Set your desired width in pixels
    height=1100  # Set your desired height in pixels
)
fig.update_xaxes(title_font_family="Arial")
fig.show()
fig.update_xaxes(title_font_family="Arial")
fig.show()
# Save fig with DPI=300
fig.write_image("study2_fig_a.png",  scale=3)

a_lcoh_0, a_lcoh_1, a_hcoh_0, a_hcoh_1 = vat_tms_coh.nodes_db.node[[
    'a(0)', 't(1)', 'a(0)', 'a(1)']]

print("P(a Low Coherence: Sham > TMS)=", (a_lcoh_0.trace() > a_lcoh_1.trace()).mean())
print("P(a High Coherence: Sham < TMS)=", (a_hcoh_0.trace() < a_hcoh_1.trace()).mean())

print("P(a Sham: Low > High)=", (a_lcoh_0.trace() > a_hcoh_0.trace()).mean())
print("P(a TMS: Low > High)=", (a_lcoh_1.trace() > a_hcoh_1.trace()).mean())

#%%
set_font_size=35
#T
# Initialize figure with subplots
fig = make_subplots(
    rows=1, cols=1,
    row_heights=[1000]
)


colours={"Sham": "light blue",
         "TMS": "magenta"
         }


# V Add traces
fig.add_trace(go.Violin(x=par_comp['Condition'][ par_comp['Session'] == 'Sham' ],
                        y=par_comp['t'][ par_comp['Session'] == 'Sham' ],
                        legendgroup='pdi', scalegroup='Sham', name='Sham',
                        legendgrouptitle_text="Session", showlegend=False,
                        line_color=colours['Sham']), col=1, row=1)


fig.add_trace(go.Violin(x=par_comp['Condition'][ par_comp['Session'] == 'TMS' ],
                        y=par_comp['t'][ par_comp['Session'] == 'TMS' ],
                        legendgroup='pdi', scalegroup='TMS', name='TMS',showlegend=False,
                        line_color=colours['TMS']), col=1, row=1)

fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(violinmode='overlay')

# Add interaction line
fig.add_shape(type='line',
               x0='Low Coherence',
               y0=vat_tms_coh_traces['t(1)'].mean(),
               x1='High Coherence',
               y1=vat_tms_coh_traces['t(1)'].mean(),
               line=dict(color='magenta',dash='dash'))

# Add interaction line
fig.add_shape(type='line',
               x0='Low Coherence',
               y0=vat_tms_coh_traces['t(0)'].mean(),
               x1='High Coherence',
               y1=vat_tms_coh_traces['t(0)'].mean(),
               line=dict(color='blue',dash='dash'))

fig.add_trace(go.Scatter(
    x=['Low Coherence'],
    y=[.48],
    legendgroup="significant",
    legendgrouptitle_text="Annotation",
    name="P<0.05",
    mode="markers",showlegend=False,
    marker=dict(color="Black", symbol='star', size=15)
))

fig.add_trace(go.Scatter(
    x=['High Coherence'],
    y=[.48],
    legendgroup="significant",
    legendgrouptitle_text="Annotation",
    name="P<0.05",
    mode="markers",showlegend=False, 
    marker=dict(color="Black", symbol='star', size=15)
))


fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

# Update xaxis properties
# Update xaxis properties
fig.update_xaxes(title_text="Motion coherence", col=1, row=1)

# Update yaxis properties
fig.update_yaxes(title_text="Non-decision time",range=[0, 0.55], col=1, row=1)


# Update title and height
fig.update_xaxes(title_font_family="Arial")

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=set_font_size,
    title_font_family="Times New Roman",
    title_font_color="black",
    legend_title_font_color="blue",
    legend_font_size=set_font_size,
    legend=dict(
        orientation="h",    
        entrywidth=300,
        yanchor="bottom",
        y=0.8,
        xanchor="right",
        x=2.5
    ), width=1200,  # Set your desired width in pixels
    height=1100  # Set your desired height in pixels
)
fig.update_xaxes(title_font_family="Arial")
fig.show()
# Save fig with DPI=300
fig.write_image("study2_fig_t.png",  scale=3)

t_lcoh_0, t_lcoh_1, t_hcoh_0, t_hcoh_1 = vat_tms_coh.nodes_db.node[[
    't(0)', 't(1)', 't(0)', 't(1)']]

print("P(t Low Coherence: Sham > TMS)=", (t_lcoh_0.trace() > t_lcoh_1.trace()).mean())
print("P(t High Coherence: Sham < TMS)=", (t_hcoh_0.trace() < t_hcoh_1.trace()).mean())

print("P(t Sham: Low > High)=", (t_lcoh_0.trace() > t_hcoh_0.trace()).mean())
print("P(t TMS: Low > High)=", (t_lcoh_1.trace() > t_hcoh_1.trace()).mean())


#%%
########
#Figures

#Get traces of parameters for plotting



vat_tms_pdi= hddm.load('vat_tms_pdi')
vat_tms_pdi.print_stats()
vat_tms_caps= hddm.load('vat_tms_caps')
vat_tms_caps.print_stats()

vat_tms_pdi_traces = vat_tms_pdi.get_traces().assign(Schizotipy = "Delusion-like")
vat_tms_caps_traces = vat_tms_caps.get_traces().assign(Schizotipy = "Hallucination-like")


###preparing dataset for plotting

#PDI
par_est_sham_lowpdi=vat_tms_pdi_traces[['a(low_pdi.0.0)','v(low_pdi.0.0)','t(low_pdi.0.0)','Schizotipy']].assign(Session='Sham',
    Group = "Low PDI")
par_est_sham_lowpdi.rename(columns = {'a(low_pdi.0.0)':'a','v(low_pdi.0.0)':'v','t(low_pdi.0.0)':'t'}, inplace = True)


par_est_sham_highpdi=vat_tms_pdi_traces[['a(high_pdi.0.0)','v(high_pdi.0.0)','t(high_pdi.0.0)','Schizotipy']].assign(Session='Sham',
    Group = "High PDI")
par_est_sham_highpdi.rename(columns = {'a(high_pdi.0.0)':'a','v(high_pdi.0.0)':'v','t(high_pdi.0.0)':'t'}, inplace = True)


par_est_tms_lowpdi=vat_tms_pdi_traces[['a(low_pdi.1.0)','v(low_pdi.1.0)','t(low_pdi.1.0)','Schizotipy']].assign(Session='TMS',
    Group = "Low PDI")
par_est_tms_lowpdi.rename(columns = {'a(low_pdi.1.0)':'a','v(low_pdi.1.0)':'v','t(low_pdi.1.0)':'t'}, inplace = True)


par_est_tms_highpdi=vat_tms_pdi_traces[['a(high_pdi.1.0)','v(high_pdi.1.0)','t(high_pdi.1.0)','Schizotipy']].assign(Session='TMS',
    Group = "High PDI")
par_est_tms_highpdi.rename(columns = {'a(high_pdi.1.0)':'a','v(high_pdi.1.0)':'v','t(high_pdi.1.0)':'t'}, inplace = True)

par_pdi=pd.concat([par_est_sham_lowpdi, par_est_sham_highpdi,
                  par_est_tms_lowpdi,par_est_tms_highpdi])

#CAPS
par_est_sham_lowcaps=vat_tms_caps_traces[['a(low_caps.0.0)','v(low_caps.0.0)','t(low_caps.0.0)','Schizotipy']].assign(Session='Sham',
    Group = "Low CAPS")
par_est_sham_lowcaps.rename(columns = {'a(low_caps.0.0)':'a','v(low_caps.0.0)':'v','t(low_caps.0.0)':'t'}, inplace = True)


par_est_sham_highcaps=vat_tms_caps_traces[['a(high_caps.0.0)','v(high_caps.0.0)','t(high_caps.0.0)','Schizotipy']].assign(Session='Sham',
    Group = "High CAPS")
par_est_sham_highcaps.rename(columns = {'a(high_caps.0.0)':'a','v(high_caps.0.0)':'v','t(high_caps.0.0)':'t'}, inplace = True)


par_est_tms_lowcaps=vat_tms_caps_traces[['a(low_caps.1.0)','v(low_caps.1.0)','t(low_caps.1.0)','Schizotipy']].assign(Session='TMS',
    Group = "Low CAPS")
par_est_tms_lowcaps.rename(columns = {'a(low_caps.1.0)':'a','v(low_caps.1.0)':'v','t(low_caps.1.0)':'t'}, inplace = True)


par_est_tms_highcaps=vat_tms_caps_traces[['a(high_caps.1.0)','v(high_caps.1.0)','t(high_caps.1.0)','Schizotipy']].assign(Session='TMS',
    Group = "High CAPS")
par_est_tms_highcaps.rename(columns = {'a(high_caps.1.0)':'a','v(high_caps.1.0)':'v','t(high_caps.1.0)':'t'}, inplace = True)



par_caps=pd.concat([par_est_sham_lowcaps, par_est_sham_highcaps,
                  par_est_tms_lowcaps,par_est_tms_highcaps])

par_comp=pd.concat([par_pdi, par_caps])
# par_est_a=par_est[['a','Session','Condition']].assign(Parameter='Decision threshold')
# par_est_a.rename(columns = {'a':'Estimate'}, inplace = True)

# par_est_v=par_est[['v','Session','Condition']].assign(Parameter='Drift rate')
# par_est_v.rename(columns = {'v':'Estimate'}, inplace = True)

# par_est_t=par_est[['t','Session','Condition']].assign(Parameter='Non-decision time')
# par_est_t.rename(columns = {'t':'Estimate'}, inplace = True)
#par_comp=pd.concat([par_est_a, par_est_v,par_est_t])


#%%
#####

#V
# Initialize figure with subplots
fig = make_subplots(
    rows=1, cols=2
)

coloursdist={
         "PDI": "rgb(135, 197, 95)",
         "CAPS": "rgb(248, 156, 116)",
         "TMS": "red",
         "TMSxPDI": "mediumseagreen",
         "TMSxCAPS": "orange"}
opac=0.6
# Add traces
fig.add_trace(go.Violin(x=par_pdi['Group'][ par_pdi['Session'] == 'Sham' ],
                        y=par_pdi['v'][ par_pdi['Session'] == 'Sham' ],
                        legendgroup='Sham', scalegroup='Sham', name='Sham',
                        line_color='blue', opacity=opac), col=1, row=1)
fig.add_trace(go.Violin(x=par_pdi['Group'][ par_pdi['Session'] == 'TMS' ],
                        y=par_pdi['v'][ par_pdi['Session'] == 'TMS' ],
                        legendgroup='TMS', scalegroup='TMS', name='TMS',
                        line_color='red',opacity=opac), col=1, row=1)
fig.update_traces(showlegend=False)

fig.add_trace(go.Violin(x=par_caps['Group'][ par_caps['Session'] == 'Sham' ],
                        y=par_caps['v'][ par_caps['Session'] == 'Sham' ],
                        legendgroup='Sham', scalegroup='Sham', name='Sham',
                        line_color='blue',opacity=opac), col=2, row=1)
fig.add_trace(go.Violin(x=par_caps['Group'][ par_caps['Session'] == 'TMS' ],
                        y=par_caps['v'][ par_caps['Session'] == 'TMS' ],
                        legendgroup='TMS', scalegroup='TMS', name='TMS',
                        line_color='red',opacity=opac), col=2, row=1)

fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(violinmode='overlay')

#add line to show interaction
fig.add_shape(type='line',
                x0='Low PDI',
                y0=vat_tms_pdi_traces['v(low_pdi.0.0)'].mean(),
                x1='High PDI',
                y1=vat_tms_pdi_traces['v(high_pdi.0.0)'].mean(),
                line=dict(color='blue',),opacity=opac,
                row=1,
                col=1)


fig.add_shape(type='line',
                x0='Low PDI',
                y0=vat_tms_pdi_traces['v(low_pdi.1.0)'].mean(),
                x1='High PDI',
                y1=vat_tms_pdi_traces['v(high_pdi.1.0)'].mean(),
                line=dict(color='red',),opacity=opac,
                row=1,
                col=1)  

fig.add_shape(type='line',
                x0='Low CAPS',
                y0=vat_tms_caps_traces['v(low_caps.0.0)'].mean(),
                x1='High CAPS',
                y1=vat_tms_caps_traces['v(high_caps.0.0)'].mean(),
                line=dict(color='blue',),opacity=opac,
                row=1,
                col=2)


fig.add_shape(type='line',
                x0='Low CAPS',
                y0=vat_tms_caps_traces['v(low_caps.1.0)'].mean(),
                x1='High CAPS',
                y1=vat_tms_caps_traces['v(high_caps.1.0)'].mean(),
                line=dict(color='red',),opacity=opac,
                row=1,
                col=2)  
# Update xaxis properties
fig.update_xaxes(title_text="PDI group", col=1, row=1)
fig.update_xaxes(title_text="CAPS group", col=2, row=1)

# Update yaxis properties
fig.update_yaxes(title_text="Drift rate",range=[0.2, 1.1], col=1, row=1)
fig.update_yaxes(title_text="Drift rate",range=[0.2, 1.1], col=2, row=1)


v_lpdi_0, v_lpdi_1, v_hpdi_0, v_hpdi_1 = vat_tms_pdi.nodes_db.node[[
    'v(low_pdi.0.0)', 'v(low_pdi.1.0)', 'v(high_pdi.0.0)', 'v(high_pdi.1.0)']]

print("P(v Low PDI: Sham <  TMS)=",
      (v_lpdi_0.trace() < v_lpdi_1.trace()).mean())
print("P(v High PDI: Sham < TMS)=",
      (v_hpdi_0.trace() < v_hpdi_1.trace()).mean())


print("P(v Sham: Low <  High)=",
      (v_lpdi_0.trace() < v_hpdi_0.trace()).mean())
print("P(v TMS: Low < High)=",
      (v_lpdi_1.trace() < v_hpdi_1.trace()).mean())


#annotations
fig.add_annotation(
    x='High PDI'
    , y=1.5
    , text='<b>**<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace"),col=1, row=1)


v_lcaps_0, v_lcaps_1, v_hcaps_0, v_hcaps_1 = vat_tms_caps.nodes_db.node[[
    'v(low_caps.0.0)', 'v(low_caps.1.0)', 'v(high_caps.0.0)', 'v(high_caps.1.0)']]

print("P(v Low PDI: Sham <  TMS)=",
      (v_lcaps_0.trace() < v_lcaps_1.trace()).mean())
print("P(v High PDI: Sham < TMS)=",
      (v_hcaps_0.trace() < v_hcaps_1.trace()).mean())


print("P(v Sham: Low <  High)=",
      (v_lcaps_0.trace() < v_hcaps_0.trace()).mean())
print("P(v TMS: Low < High)=",
      (v_lcaps_1.trace() < v_hcaps_1.trace()).mean())


#annotations
fig.add_annotation(
    x='High CAPS'
    , y=1.5
    , text='<b>P=0.05<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace"),col=1, row=1)


# Update title and height
fig.update_xaxes(title_font_family="Arial")

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=18,
    title_font_family="Times New Roman",
    title="A.",
    title_font_color="black",  
    legend_title_font_color="blue",
    legend_font_size=22,width=1300, height=800
)

fig.show()
#%%

#A
# Initialize figure with subplots
fig = make_subplots(
    rows=1, cols=2
)

# Add traces
fig.add_trace(go.Violin(x=par_pdi['Group'][ par_pdi['Session'] == 'Sham' ],
                        y=par_pdi['a'][ par_pdi['Session'] == 'Sham' ],
                        legendgroup='Sham', scalegroup='Sham', name='Sham',
                        line_color='blue',opacity=opac), col=1, row=1)
fig.add_trace(go.Violin(x=par_pdi['Group'][ par_pdi['Session'] == 'TMS' ],
                        y=par_pdi['a'][ par_pdi['Session'] == 'TMS' ],
                        legendgroup='TMS', scalegroup='TMS', name='TMS',
                        line_color='red',opacity=opac), col=1, row=1)
fig.update_traces(showlegend=False)
fig.add_trace(go.Violin(x=par_caps['Group'][ par_caps['Session'] == 'Sham' ],
                        y=par_caps['a'][ par_caps['Session'] == 'Sham' ],
                        legendgroup='Sham', scalegroup='Sham', name='Sham',
                        line_color='blue',opacity=opac), col=2, row=1)
fig.add_trace(go.Violin(x=par_caps['Group'][ par_caps['Session'] == 'TMS' ],
                        y=par_caps['a'][ par_caps['Session'] == 'TMS' ],
                        legendgroup='TMS', scalegroup='TMS', name='TMS',
                        line_color='red',opacity=opac), col=2, row=1)

fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(violinmode='overlay')
  
# Update xaxis properties
fig.update_xaxes(title_text="PDI group", col=1, row=1)
fig.update_xaxes(title_text="CAPS group", col=2, row=1)

# Update yaxis properties
fig.update_yaxes(title_text="Decision threshold",range=[2, 3.5], col=1, row=1)
fig.update_yaxes(title_text="Decision threshold",range=[2, 3.5], col=2, row=1)


#add line to show interaction
fig.add_shape(type='line',
                x0='Low PDI',
                y0=vat_tms_pdi_traces['a(low_pdi.0.0)'].mean(),
                x1='High PDI',
                y1=vat_tms_pdi_traces['a(high_pdi.0.0)'].mean(),
                line=dict(color='blue',),opacity=opac,
                row=1,
                col=1)


fig.add_shape(type='line',
                x0='Low PDI',
                y0=vat_tms_pdi_traces['a(low_pdi.1.0)'].mean(),
                x1='High PDI',
                y1=vat_tms_pdi_traces['a(high_pdi.1.0)'].mean(),
                line=dict(color='red',),opacity=opac,
                row=1,
                col=1)  

fig.add_shape(type='line',
                x0='Low CAPS',
                y0=vat_tms_caps_traces['a(low_caps.0.0)'].mean(),
                x1='High CAPS',
                y1=vat_tms_caps_traces['a(high_caps.0.0)'].mean(),
                line=dict(color='blue',),opacity=opac,
                row=1,
                col=2)


fig.add_shape(type='line',
                x0='Low CAPS',
                y0=vat_tms_caps_traces['a(low_caps.1.0)'].mean(),
                x1='High CAPS',
                y1=vat_tms_caps_traces['a(high_caps.1.0)'].mean(),
                line=dict(color='red',),
                row=1,
                col=2)  

#annotations
fig.add_annotation(
    x='Low PDI'
    , y=5.1
    , text='<b>*<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace"),col=1, row=1)

fig.add_annotation(
    x='High PDI'
    , y=4.9
    , text='<b>*<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace"),col=1, row=1)

fig.add_annotation(
    x='Low CAPS'
    , y=5.4
    , text='<b>***<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace"),col=2, row=1)

# Update title and height
fig.update_xaxes(title_font_family="Arial")

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=18,
    title_font_family="Times New Roman",
    title="B.",
    title_font_color="black",  
    legend_title_font_color="blue",
    legend_font_size=22,width=1300, height=800
)

fig.show()
#%%


#T
# Initialize figure with subplots
fig = make_subplots(
    rows=1, cols=2
)

# Add traces
fig.add_trace(go.Violin(x=par_pdi['Group'][ par_pdi['Session'] == 'Sham' ],
                        y=par_pdi['t'][ par_pdi['Session'] == 'Sham' ],
                        legendgroup='Sham', scalegroup='Sham', name='Sham',
                        line_color='blue',opacity=opac), col=1, row=1)
fig.add_trace(go.Violin(x=par_pdi['Group'][ par_pdi['Session'] == 'TMS' ],
                        y=par_pdi['t'][ par_pdi['Session'] == 'TMS' ],
                        legendgroup='TMS', scalegroup='TMS', name='TMS',
                        line_color='red',opacity=opac), col=1, row=1)
fig.update_traces(showlegend=False)
fig.add_trace(go.Violin(x=par_caps['Group'][ par_caps['Session'] == 'Sham' ],
                        y=par_caps['t'][ par_caps['Session'] == 'Sham' ],
                        legendgroup='Sham', scalegroup='Sham', name='Sham',
                        line_color='blue',opacity=opac), col=2, row=1)
fig.add_trace(go.Violin(x=par_caps['Group'][ par_caps['Session'] == 'TMS' ],
                        y=par_caps['t'][ par_caps['Session'] == 'TMS' ],
                        legendgroup='TMS', scalegroup='TMS', name='TMS',
                        line_color='red',opacity=opac), col=2, row=1)

fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(violinmode='overlay')



#add line to show interaction
fig.add_shape(type='line',
                x0='Low PDI',
                y0=vat_tms_pdi_traces['t(low_pdi.0.0)'].mean(),
                x1='High PDI',
                y1=vat_tms_pdi_traces['t(high_pdi.0.0)'].mean(),
                line=dict(color='blue',),opacity=opac,
                row=1,
                col=1)


fig.add_shape(type='line',
                x0='Low PDI',
                y0=vat_tms_pdi_traces['t(low_pdi.1.0)'].mean(),
                x1='High PDI',
                y1=vat_tms_pdi_traces['t(high_pdi.1.0)'].mean(),
                line=dict(color='red',),opacity=opac,
                row=1,
                col=1)  

fig.add_shape(type='line',
                x0='Low CAPS',
                y0=vat_tms_caps_traces['t(low_caps.0.0)'].mean(),
                x1='High CAPS',
                y1=vat_tms_caps_traces['t(high_caps.0.0)'].mean(),
                line=dict(color='blue',),opacity=opac,
                row=1,
                col=2)


fig.add_shape(type='line',
                x0='Low CAPS',
                y0=vat_tms_caps_traces['t(low_caps.1.0)'].mean(),
                x1='High CAPS',
                y1=vat_tms_caps_traces['t(high_caps.1.0)'].mean(),
                line=dict(color='red',),opacity=opac,
                row=1,
                col=2)  
  
# Update xaxis properties
fig.update_xaxes(title_text="PDI group", col=1, row=1)
fig.update_xaxes(title_text="CAPS group", col=2, row=1)

# Update yaxis properties
fig.update_yaxes(title_text="Non-decision time",range=[0.1, 0.6], col=1, row=1)
fig.update_yaxes(title_text="Non-decision time",range=[0.1, 0.6], col=2, row=1)

# Update title and height
fig.update_xaxes(title_font_family="Arial")

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=18,
    title_font_family="Times New Roman",
    title="C.",
    title_font_color="black",  
    legend_title_font_color="blue",
    legend_font_size=22,width=1300, height=800
)

fig.show()
#%%





###########################################
####### Signle violin plots


colours={"Low PDI": "rgb(139, 224, 164)",
         "High PDI": "rgb(135, 197, 95)",
         "Low CAPS": "rgb(246, 207, 113)",
         "High CAPS": "rgb(248, 156, 116)",
         "Low ASI": "rgb(220, 176, 242)",
         "High ASI": "rgb(180, 151, 231)"}


#v

fig = px.violin(par_pdi, y="v", x="Group", box=True, color='Session',  
                facet_col='Schizotipy',
                violinmode='overlay',
                hover_data=par_pdi.columns,
                labels= {'v':'Drift rate'},
                width=1300,
                height=800)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=18,
    title_font_family="Times New Roman",
    title="A.",
    title_font_color="black",    legend_title_font_color="blue",
    legend_font_size=18
)
fig.update_xaxes(title_font_family="Arial")

fig.add_annotation(
    x='High PDI'
    , y=1.5
    , text='<b>**<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace"))


fig.show()

#a

fig = px.violin(par_pdi, y="a", x="Group", box=True, color='Session',  
                violinmode='overlay',
                hover_data=par_pdi.columns,
                labels= {'a':'Decision threshold'},
                width=1300,
                height=800)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=18,
    title_font_family="Times New Roman",
    title="B.",
    title_font_color="black",    legend_title_font_color="blue",
    legend_font_size=18
)
fig.update_xaxes(title_font_family="Arial")

fig.add_annotation(
    x='Low PDI'
    , y=5.1
    , text='<b>**<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace"))


fig.add_annotation(
    x='High PDI'
    , y=4.9
    , text='<b>**<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace"))


fig.show()

#t

fig = px.violin(par_pdi, y="t", x="Group", box=True, color='Session',  
                violinmode='overlay',
                hover_data=par_pdi.columns,
                labels= {'t':'Non-decision time'},
                width=1300,
                height=800)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=18,
    title_font_family="Times New Roman",
    title="C.",
    title_font_color="black",    legend_title_font_color="blue",
    legend_font_size=18
)
fig.update_xaxes(title_font_family="Arial")

fig.show()

#%%


##################
#######CAPS
#v

fig = px.violin(par_caps, y="v", x="Group", box=True, color='Session',  
                violinmode='overlay',
                hover_data=par_caps.columns,
                labels= {'v':'Drift rate'},
                width=1300,
                height=800)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=18,
    title_font_family="Times New Roman",
    title="A.",
    title_font_color="black",    legend_title_font_color="blue",
    legend_font_size=18
)
fig.update_xaxes(title_font_family="Arial")


fig.show()

#a

fig = px.violin(par_caps, y="a", x="Group", box=True, color='Session',  
                violinmode='overlay',
                hover_data=par_caps.columns,
                labels= {'a':'Decision threshold'},
                width=1300,
                height=800)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=18,
    title_font_family="Times New Roman",
    title="B.",
    title_font_color="black",    legend_title_font_color="blue",
    legend_font_size=18
)
fig.update_xaxes(title_font_family="Arial")

fig.add_annotation(
    x='Low CAPS'
    , y=5.4
    , text='<b>**<b>'
    ,showarrow=False
    , font=dict(size=18, color="black", family="Courier New, monospace"))


fig.show()

#t

fig = px.violin(par_caps, y="t", x="Group", box=True, color='Session',  
                violinmode='overlay',
                hover_data=par_caps.columns,
                labels= {'t':'Non-decision time'},
                width=1300,
                height=800)

fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

fig.update_layout(
    font_family="Courier New",
    font_color="blue",
    font_size=18,
    title_font_family="Times New Roman",
    title="C.",
    title_font_color="black",    legend_title_font_color="blue",
    legend_font_size=18
)
fig.update_xaxes(title_font_family="Arial")

fig.show()
