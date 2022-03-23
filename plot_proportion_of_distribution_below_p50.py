#!/usr/bin/env python

"""
Plot the proportion of each species' distribution that is below p50 for each of
the experiments (CTL, rPPT and eCO2)

NB. rPPT is called ePPT in the output file, but renamed for the plots.

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (23.03.2022)"
__email__ = "mdekauwe@gmail.com"

import xarray as xr
import matplotlib.pyplot as plt
import sys
import datetime as dt
import pandas as pd
import numpy as np
from matplotlib.ticker import FixedLocator
import datetime
import os
import glob
import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")

pal = sns.color_palette("pastel").as_hex()

def main(fname, fname2):

    df = pd.read_csv(fname)
    df = df[df.days > -500.].reset_index() # badly named, it is months...
    df['map'] = np.ones(len(df)) * np.nan

    species_list = ['Eucalyptus largiflorens','Eucalyptus populnea',\
                    'Eucalyptus blakelyi','Eucalyptus obliqua',\
                    'Eucalyptus camaldulensis','Eucalyptus sideroxylon',\
                    'Eucalyptus macrorhyncha','Eucalyptus viminalis',\
                    'Eucalyptus melliodora','Eucalyptus globulus',\
                    'Eucalyptus tereticornis','Eucalyptus saligna',\
                    'Eucalyptus crebra','Eucalyptus grandis',\
                    'Eucalyptus dunnii']
    vals, vals_co2 = [], []
    for spp in species_list:

        ctl = df[(df.species == spp) & \
                 (df.experiment == "control")].reset_index()
        eppt = df[(df.species == spp) & \
                  (df.experiment == "ePPT")].reset_index()
        eco2_eppt = df[(df.species == spp) & \
                       (df.experiment == "eCO2_ePPT")].reset_index()

        #ctl_med = np.median(ctl.plc_max.values)
        #eppt_med = np.median(eppt.plc_max.values)
        #eco2_eppt_med = np.median(eco2_eppt.plc_max.values)

        ctl_med = np.mean(ctl.days.values)
        eppt_med = np.mean(eppt.days.values)
        eco2_eppt_med = np.mean(eco2_eppt.days.values)

        if ~np.isnan(eppt_med):

            change = (ctl_med-eppt_med) / ctl_med * 100.
            vals.append(change)
            #print("ePPT ... %s: %.2f -- %.2f, %.2f, %.2f" % (spp, change, ctl_med, eppt_med, eco2_eppt_med))

        if ~np.isnan(eco2_eppt_med):
            change_co2 = (eppt_med-eco2_eppt_med) / eppt_med * 100.
            vals_co2.append(change_co2)

    vals = np.array(vals)
    vals_co2 = np.array(vals_co2)

    df_map = pd.read_csv("euc_map_east.csv")
    df_map = df_map.sort_values(by=['map'])
    #print(df_map)
    species = df_map.species
    species = species.str.replace("_", " ")
    species = species.str.replace("Eucalyptus", "E.")

    df2 = pd.read_csv(fname2)
    df2['map'] = np.ones(len(df2)) * np.nan

    for i in range(len(df_map)):
        spp = df_map.species[i]
        map = df_map.map[i]

        for j in range(len(df)):

            if df.species[j] == spp.replace(" ", "_"):
                df['map'][j] = map

        for k in range(len(df2)):
            if df2.species[k] == spp.replace(" ", "_"):

                df2['map'][k] = map

    for i in range(len(df_map)):
        spp = df_map.species[i]
        map = df_map.map[i]

        for k in range(len(df2)):
            if df2.species[k] == spp:
                df2['map'][k] = map
    df2 = df2.sort_values(by=['map'])
    sorted_map = df_map['map'].values
    #df = df.sort_values(by=['map'])

    #for i in range(len(df)):
    #    print(i, df.species[i], df.psi_leaf_mean[i])

    df['species'] = df['species'].str.replace("_", " ")
    df['species'] = df['species'].str.replace("Eucalyptus", "E.")

    df2['species'] = df2['species'].str.replace("_", " ")
    df2['species'] = df2['species'].str.replace("Eucalyptus", "E.")
    #species = np.unique(df.species)


    sns.set_style("ticks")
    sns.set_style({"xtick.direction": "in","ytick.direction": "in"})

    fig, axs = plt.subplots(5, 3, figsize=(9,8))
    fig.subplots_adjust(hspace=0.15)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    row_cnt = 0
    col_cnt = 0

    #species = np.unique(df.species)
    species = ['E. largiflorens','E. camaldulensis',\
               'E. populnea','E. sideroxylon',\
               'E. melliodora','E. blakelyi',\
               'E. macrorhyncha','E. viminalis',\
               'E. crebra', 'E. obliqua',\
               'E. globulus', 'E. tereticornis',\
               'E. saligna','E. dunnii','E. grandis']

    spp_cnt = 0
    for i in range(axs.shape[0]):
        for j in range(0, axs.shape[1]):

            spp = species[spp_cnt]

            ctl_zero = df[(df.species == spp) & \
                     (df.experiment == "control") & (df.days == 0)].reset_index()
            eppt_zero = df[(df.species == spp) & \
                      (df.experiment == "ePPT") & (df.days == 0)].reset_index()
            eco2_eppt_zero = df[(df.species == spp) & \
                           (df.experiment == "eCO2_ePPT") & (df.days == 0)].reset_index()

            ctl = df[(df.species == spp) & \
                     (df.experiment == "control") & (df.days > 0)].reset_index()
            eppt = df[(df.species == spp) & \
                      (df.experiment == "ePPT") & (df.days > 0)].reset_index()
            eco2_eppt = df[(df.species == spp) & \
                           (df.experiment == "eCO2_ePPT") & (df.days > 0)].reset_index()


            total_ctl = len(ctl_zero) + len(ctl)
            total_eppt = len(eppt_zero) + len(eppt)
            total_eco2_eppt = len(eco2_eppt_zero) + len(eco2_eppt)


            df_bar = pd.DataFrame({'species' : [], 'experiment' : [],
                                    'Proportion of distribution (%)' : []})

            if total_ctl > 0:
                new_row = {'species':spp, 'experiment':"CTL",
                            'Proportion of distribution (%)':len(ctl) / total_ctl * 100.}
            else:
                new_row = {'species':spp, 'experiment':"CTL",
                            'Proportion of distribution (%)':0.0}
            df_bar = df_bar.append(new_row, ignore_index=True)
            if total_eppt > 0:
                new_row = {'species':spp, 'experiment':"rPPT",
                            'Proportion of distribution (%)':len(eppt) / total_eppt * 100.}
            else:
                new_row = {'species':spp, 'experiment':"rPPT",
                            'Proportion of distribution (%)':0.0}
            df_bar = df_bar.append(new_row, ignore_index=True)
            if total_eco2_eppt > 0:
                new_row = {'species':spp, 'experiment':"eCO$_2$ x rPPT",
                            'Proportion of distribution (%)':len(eco2_eppt) / total_eco2_eppt * 100.}
            else:
                new_row = {'species':spp, 'experiment':"eCO$_2$ x rPPT",
                            'Proportion of distribution (%)':0.0}
            df_bar = df_bar.append(new_row, ignore_index=True)


            #print(df_bar)
            if total_ctl > 0 and total_eppt > 0:
                xx = len(ctl) / total_ctl * 100.
                yy = len(eppt) / total_eppt * 100.
                #print(spp, xx, yy, xx-yy)
                print(spp,  xx-yy, (yy/xx)-1)

            #sns.histplot(data=ctl, x="days", kde=True, color="skyblue", bins=36, ax=axs[i][j])
            #sns.histplot(data=eppt, x="days", kde=True, color="olive", bins=36, ax=axs[i][j])
            #sns.histplot(data=eco2_eppt, x="days", kde=True, color="teal", bins=36, ax=axs[i][j])

            #hue_order = ['control', 'ePPT', 'eCO2_ePPT']
            g1 = sns.barplot(x='experiment', y='Proportion of distribution (%)', data=df_bar,
                            palette=["#ECDE38", "#00B9F1", "#00A875"], ax=axs[i][j])
            g1.set(ylim=(0, 110))
            g1.set_yticks([0, 25, 50, 75, 100]) # <--- set the ticks first
            g1.set_yticklabels(['0','25','50','75','100'])

            if spp_cnt < 12:
                g1.set(xticklabels=[])
                g1.set(xlabel=None)
            else:
                g1.set(xlabel=None)

            if spp_cnt != 6:
                g1.set(ylabel=None)

            if spp_cnt == 1:
                g1.set(yticklabels=[])
            elif spp_cnt == 2:
                g1.set(yticklabels=[])
            elif spp_cnt == 4:
                g1.set(yticklabels=[])
            elif spp_cnt == 5:
                g1.set(yticklabels=[])
            elif spp_cnt == 7:
                g1.set(yticklabels=[])
            elif spp_cnt == 8:
                g1.set(yticklabels=[])
            elif spp_cnt == 10:
                g1.set(yticklabels=[])
            elif spp_cnt == 11:
                g1.set(yticklabels=[])
            elif spp_cnt == 13:
                g1.set(yticklabels=[])
            elif spp_cnt == 14:
                g1.set(yticklabels=[])


            axs[i][j].text(0.96, 0.82, '$\it{%s}$' % (spp),
                    horizontalalignment='right', size=10, color="black",
                    transform=axs[i][j].transAxes)
            axs[i][j].tick_params(bottom=False)

            #change_width(axs[i][j], .35)

            spp_cnt += 1

    #plt.show()


    of = "/Users/mdekauwe/Dropbox/Documents/papers/Future_euc_drought_paper/figures/figs/months_below_p50.pdf"
    #of = "/Users/mdekauwe/Desktop/proportion_distribution_below_p50.pdf"
    fig.savefig(of, bbox_inches='tight', pad_inches=0.1)

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)


if __name__ == "__main__":

    fname = "days_below_p50_all_experiments.csv"
    fname2 = "euc_species_traits_east.csv"
    main(fname, fname2)
