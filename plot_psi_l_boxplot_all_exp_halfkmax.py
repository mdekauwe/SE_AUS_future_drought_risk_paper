#!/usr/bin/env python

"""
Plot the minimum water potential across each of the drought experiments
(CTL, rPPT and eCO2). This recreates figure 2, but halving kmax

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
    df = df[df.psi_leaf_min > -500.].reset_index()
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

        ctl_med = np.mean(ctl.psi_leaf_min.values)
        eppt_med = np.mean(eppt.psi_leaf_min.values)
        eco2_eppt_med = np.mean(eco2_eppt.psi_leaf_min.values)

        if ~np.isnan(eppt_med):

            change = (ctl_med-eppt_med) / ctl_med * 100.
            vals.append(change)
            #print("ePPT ... %s: %.2f -- %.2f, %.2f, %.2f" % (spp, change, ctl_med, eppt_med, eco2_eppt_med))

        if ~np.isnan(eco2_eppt_med):
            change_co2 = (eppt_med-eco2_eppt_med) / eppt_med * 100.
            vals_co2.append(change_co2)

    vals = np.array(vals)
    vals_co2 = np.array(vals_co2)
    #print(" ")

    #print("%.2f -- (%.2f, %.2f)" % (np.mean(vals), np.min(vals), np.max(vals)))
    #print("%.2f -- (%.2f, %.2f)" % (np.mean(vals_co2), np.min(vals_co2), np.max(vals_co2)))

    ##
    # Get the matching MAP for each distribution
    ##
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

    fig = plt.figure(figsize=(12,6))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12

    flierprops = dict(marker='o', markersize=0.1, markerfacecolor="grey",
                      markeredgecolor='grey')

    #PROPS = {
    #    'boxprops':{'edgecolor':'lightgrey'},
    #    'medianprops':{'color':'lightgrey'},
    #    'whiskerprops':{'color':'lightgrey'},
    #    'capprops':{'color':'lightgrey'}
    #}

    hue_order = ['control', 'ePPT', 'eCO2_ePPT']
    ax = sns.boxplot(x="species", y="psi_leaf_min", data=df,
                     #flierprops=flierprops, palette=["m", "g", "b"],
                     flierprops=flierprops,
                     palette=["#ECDE38", "#00B9F1", "#00A875"],
                     hue="experiment", order=species, zorder=1,
                     hue_order=hue_order)
    #                 hue_order=hue_order, **PROPS)


    ax = sns.scatterplot(data=df2, x="species", y="p50", color="red", edgecolor="red",
                         marker="*", s=100, label="p$_{\mathrm{50}}$", zorder=2)



    ax.set_ylabel("$\Psi$$_{\mathrm{min}}$ (MPa)")
    ax.set_xlabel(" ")
    ax.legend(numpoints=1, loc="lower right", frameon=False, ncol=2)
    ax.set_xticklabels(species, rotation=90)


    ##8da0cb
    plt.text(-1.4, 0.5,
             "MAP\n(mm yr$^{-1}$)", horizontalalignment='center', size=10,
             color="black", weight="bold")

    offset = 0
    for i,val in enumerate(sorted_map):

        plt.text(-0.05+offset, 0.5,
                 "%d" % (val), horizontalalignment='center', size=10,
                 color="black")

        offset += 1.0

    #ax.legend(numpoints=1, loc="best", frameon=False)


    #handles, labels = ax.get_legend_handles_labels()
    #print(handles)
    #print(labels)
    #hh = handles[0:3]
    #ll = labels[0:3]
    #hh2 = handles[9:]
    #ll2 = labels[9:]

    #l = plt.legend(hh+hh2, ll+ll2, loc="best", frameon=False)

    leg = ax.get_legend()
    new_labels = ['CTL', 'rPPT', 'eCO$_2$ x rPPT']
    for t, l in zip(leg.texts, new_labels): t.set_text(l)

    #ax.set_ylim(-9.8, 0.5)

    of = "/Users/mdekauwe/Dropbox/Documents/papers/Future_euc_drought_paper/figures/figs/psi_leaf_all_years_halfkmax.pdf"
    #of = "/Users/mdekauwe/Desktop/psi_leaf_all_years_halfkmax.pdf"

    plt.savefig(of, bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

    fname = "psi_leaf_min_all_experiments_halfkmax.csv"
    fname2 = "euc_species_traits_east.csv"
    main(fname, fname2)
