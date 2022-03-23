#!/usr/bin/env python

"""
Plot midday gs vs psi_leaf for a series of experiments - changing kcrit, kmax
LAI and eCO2. Also plot matching PLC.

All simulations are from a location near Armidale, New South Wales
(30.4degS, 151.6degE).

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (23.03.2022)"
__email__ = "mdekauwe@gmail.com"

import netCDF4 as nc
import matplotlib.pyplot as plt
import sys
import datetime as dt
import pandas as pd
import numpy as np
from matplotlib.ticker import FixedLocator
import datetime
import os
import glob
import xarray as xr

def main(fname1, fname2, fname3, fname4, fname5, fname6, fname7, plot_fname=None):

    df1 = read_cable_file(fname1, type="CABLE")
    df2 = read_cable_file(fname2, type="CABLE")
    df3 = read_cable_file(fname3, type="CABLE")
    df4 = read_cable_file(fname4, type="CABLE")
    df5 = read_cable_file(fname5, type="CABLE")
    df6 = read_cable_file(fname6, type="CABLE")
    df7 = read_cable_file(fname7, type="CABLE")

    df1 = df1.sel(time=slice("2017-01-01", "2019-08-31"))
    df2 = df2.sel(time=slice("2017-01-01", "2019-08-31"))
    df3 = df3.sel(time=slice("2017-01-01", "2019-08-31"))
    df4 = df4.sel(time=slice("2017-01-01", "2019-08-31"))
    df5 = df5.sel(time=slice("2017-01-01", "2019-08-31"))
    df6 = df6.sel(time=slice("2017-01-01", "2019-08-31"))
    df7 = df7.sel(time=slice("2017-01-01", "2019-08-31"))

    fig = plt.figure(figsize=(10,6))
    fig.subplots_adjust(hspace=0.3)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['font.size'] = 12
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    colours = plt.cm.Set2(np.linspace(0, 1, 7))

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    # pre-dawn
    df1x = df1.sel(time=datetime.time(6))
    df2x = df2.sel(time=datetime.time(6))
    df3x = df3.sel(time=datetime.time(6))
    df4x = df4.sel(time=datetime.time(6))
    df5x = df5.sel(time=datetime.time(6))
    df6x = df6.sel(time=datetime.time(6))
    df7x = df7.sel(time=datetime.time(6))

    # midday
    df1y = df1.sel(time=datetime.time(12))
    df2y = df2.sel(time=datetime.time(12))
    df3y = df3.sel(time=datetime.time(12))
    df4y = df4.sel(time=datetime.time(12))
    df5y = df5.sel(time=datetime.time(12))
    df6y = df6.sel(time=datetime.time(12))
    df7y = df7.sel(time=datetime.time(12))


    window = 24 * 3

    axes = [ax1]


    #ax1.plot(df['gsw_sha']+df['gsw_sun'], df["psi_leaf"], c="black", marker=".",
    #         lw=1.5, ls=" ")

    gsw1 = df1y['gsw_sha']+df1y['gsw_sun'].values
    gsw2 = df2y['gsw_sha']+df2y['gsw_sun'].values
    gsw3 = df3y['gsw_sha']+df3y['gsw_sun'].values
    gsw4 = df4y['gsw_sha']+df4y['gsw_sun'].values
    gsw5 = df5y['gsw_sha']+df5y['gsw_sun'].values
    gsw6 = df6y['gsw_sha']+df6y['gsw_sun'].values
    gsw7 = df7y['gsw_sha']+df7y['gsw_sun'].values

    psi_leaf1 = df1y["psi_leaf"].values
    psi_leaf2 = df2y["psi_leaf"].values
    psi_leaf3 = df3y["psi_leaf"].values
    psi_leaf4 = df4y["psi_leaf"].values
    psi_leaf5 = df5y["psi_leaf"].values
    psi_leaf6 = df6y["psi_leaf"].values
    psi_leaf7 = df7y["psi_leaf"].values

    min_thresh=0.01

    gsw1 = np.where(gsw1>min_thresh, gsw1, np.nan)
    gsw2 = np.where(gsw2>min_thresh, gsw2, np.nan)
    gsw3 = np.where(gsw3>min_thresh, gsw3, np.nan)
    gsw4 = np.where(gsw4>min_thresh, gsw4, np.nan)
    gsw5 = np.where(gsw5>min_thresh, gsw5, np.nan)
    gsw6 = np.where(gsw6>min_thresh, gsw6, np.nan)
    gsw7 = np.where(gsw7>min_thresh, gsw7, np.nan)

    psi_leaf1 = np.where(gsw1>min_thresh, psi_leaf1, np.nan)
    psi_leaf2 = np.where(gsw2>min_thresh, psi_leaf2, np.nan)
    psi_leaf3 = np.where(gsw3>min_thresh, psi_leaf3, np.nan)
    psi_leaf4 = np.where(gsw4>min_thresh, psi_leaf4, np.nan)
    psi_leaf5 = np.where(gsw5>min_thresh, psi_leaf5, np.nan)
    psi_leaf6 = np.where(gsw6>min_thresh, psi_leaf6, np.nan)
    psi_leaf7 = np.where(gsw7>min_thresh, psi_leaf7, np.nan)


    ax1.plot(psi_leaf1, gsw1, c=colours[0], marker=".", markersize=3, alpha=0.8,
             lw=1.5, ls=" ", label="$k$$_{\mathrm{crit}}$ = 0.05 x $k$$_{\mathrm{max}}$")
    ax1.plot(psi_leaf2, gsw2, c=colours[1], marker=".", markersize=3, alpha=0.8,
             lw=1.5, ls=" ", label="$k$$_{\mathrm{crit}}$ = 0.15 x $k$$_{\mathrm{max}}$")
    ax1.plot(psi_leaf3, gsw3, c=colours[2], marker=".", markersize=3, alpha=0.8,
             lw=1.5, ls=" ", label="$k$$_{\mathrm{crit}}$ = 0.3 x $k$$_{\mathrm{max}}$")

    ax1.axvline(x=-3.15, ymin=0, ymax=0.4, ls="--", color="grey", label="P$_{50}$")
    ax1.axvline(x=-4.43, ymin=0, ymax=0.4, ls="--", color="black", label="P$_{88}$")

    ax2.plot(psi_leaf1, gsw1, c=colours[0], marker=".", markersize=3, alpha=0.8,
             lw=1.5, ls=" ", label="$k$$_{\mathrm{max}}$ = 1.5")
    ax2.plot(psi_leaf4, gsw4, c=colours[1], marker=".", markersize=3, alpha=0.8,
             lw=1.5, ls=" ", label="$k$$_{\mathrm{max}}$ = 0.75")
    #ax2.plot(psi_leaf5, gsw5, c=colours[2], marker=".", markersize=3, alpha=0.8,
    #         lw=1.5, ls=" ", label="k$_{\mathrm{max}}$ = 0.3")

    ax3.plot(psi_leaf1, gsw1, c=colours[0], marker=".", markersize=3, alpha=0.8,
             lw=1.5, ls=" ", label="aCO$_2$")
    ax3.plot(psi_leaf6, gsw6, c=colours[1], marker=".", markersize=3, alpha=0.8,
             lw=1.5, ls=" ", label="eLAI")
    ax3.plot(psi_leaf7, gsw7, c=colours[2], marker=".", markersize=3, alpha=0.8,
             lw=1.5, ls=" ", label="eCO$_2$ x eLAI")

    ax4.plot(df1.time, df1.plc, c=colours[0], marker=".",
             lw=0.1, ls="-", label="$k_{crit}$ = 0.05 x $k$$_{max}$")
    ax4.plot(df2.time, df2.plc, c=colours[1], marker=".",
             lw=0.1, ls="-", label="$k_{crit}$ = 0.15 x $k$$_{max}$")
    ax4.plot(df3.time, df3.plc, c=colours[2], marker=".",
             lw=0.1, ls="-", label="$k_{crit}$ = 0.3 x $k$$_{max}$")
    ax4.axhline(y=88, color="red", ls="--")

    ax5.plot(df1.time, df1.plc, c=colours[0], marker=".",
             lw=0.1, ls="-", label="K$_{\mathrm{max}}$ = 1.5")
    ax5.plot(df4.time, df4.plc, c=colours[1], marker=".",
             lw=0.1, ls="-", label="K$_{\mathrm{max}}$ = 0.75")
    #ax5.plot(df5.time, df5.plc, c=colours[2], marker=".",
    #         lw=0.1, ls="-", label="K$_{\mathrm{max}}$ = 0.3")
    #ax5.axhline(y=0.88, color="red", ls="--")
    ax5.axhline(y=88, color="red", ls="--")


    #ax6.plot(df1.time, df1.plc/88., c=colours[0], marker=".",
    #         lw=0.1, ls="-", label="aCO$_2$")
    #ax6.plot(df6.time, df6.plc/88., c=colours[1], marker=".",
    #         lw=0.1, ls="-", label="eLAI")
    #ax6.plot(df7.time, df7.plc/88., c=colours[2], marker=".",
    #         lw=0.1, ls="-", label="eCO$_2$ x eLAI")
    ax6.plot(df1.time, df1.plc, c=colours[0], marker=".",
             lw=0.1, ls="-", label="aCO$_2$")
    ax6.plot(df6.time, df6.plc, c=colours[1], marker=".",
             lw=0.1, ls="-", label="eLAI")
    ax6.plot(df7.time, df7.plc, c=colours[2], marker=".",
             lw=0.1, ls="-", label="eCO$_2$ x eLAI")
    ax6.axhline(y=88, color="red", ls="--")

    ax1.set_xlabel("Midday $\Psi$$_{\mathrm{leaf}}$ (MPa)")
    ax2.set_xlabel("Midday $\Psi$$_{\mathrm{leaf}}$ (MPa)")
    ax3.set_xlabel("Midday $\Psi$$_{\mathrm{leaf}}$ (MPa)")

    ax1.set_ylabel("Midday g$_{\mathrm{sw}}$ (mol m$^{-2}$ s$^{-1}$)")

    #ax4.set_ylabel("Relative PLC (-)")
    ax4.set_ylabel("PLC (%)")


    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(5))
    ax2.xaxis.set_major_locator(MaxNLocator(5))
    ax3.xaxis.set_major_locator(MaxNLocator(5))
    ax4.xaxis.set_major_locator(MaxNLocator(3))
    ax5.xaxis.set_major_locator(MaxNLocator(3))
    ax6.xaxis.set_major_locator(MaxNLocator(3))



    lg1 = ax1.legend(numpoints=1, ncol=1, loc="best", frameon=False)
    lg2 = ax2.legend(numpoints=1, loc="best", frameon=False)
    lg3 = ax3.legend(numpoints=1, loc="best", frameon=False)
    #ax3.legend(numpoints=1, loc="best", frameon=False)

    #lg1.legendHandles[0]._legmarker.set_markersize(8)
    #lg1.legendHandles[1]._legmarker.set_markersize(8)
    #lg1.legendHandles[2]._legmarker.set_markersize(8)
    #lg2.legendHandles[0]._legmarker.set_markersize(8)
    #lg2.legendHandles[1]._legmarker.set_markersize(8)
    #lg3.legendHandles[0]._legmarker.set_markersize(8)
    #lg3.legendHandles[1]._legmarker.set_markersize(8)
    #lg3.legendHandles[2]._legmarker.set_markersize(8)

    import matplotlib.dates as mdates
    myFmt = mdates.DateFormatter('%Y')
    ax4.xaxis.set_major_formatter(myFmt)
    ax5.xaxis.set_major_formatter(myFmt)
    ax6.xaxis.set_major_formatter(myFmt)

    ax1.set_xlim(0.0, -4.8)
    ax2.set_xlim(0.0, -4.8)
    ax3.set_xlim(0.0, -4.8)
    ax1.set_ylim(0.0, 0.6)
    ax2.set_ylim(0.0, 0.6)
    ax3.set_ylim(0.0, 0.6)


    #ax4.set_ylim(-0.05, 0.9)
    #ax5.set_ylim(-0.05, 0.9)
    #ax6.set_ylim(-0.05, 0.9)
    ax4.set_ylim(-5, 90)
    ax5.set_ylim(-5, 90)
    ax6.set_ylim(-5, 90)

    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax5.get_yticklabels(), visible=False)
    plt.setp(ax6.get_yticklabels(), visible=False)
    #plt.setp(ax1.get_xticklabels(), visible=False)
    #plt.setp(ax2.get_xticklabels(), visible=False)

    #for a in axes:
    #    a.set_xlim([datetime.date(2017,1,1), datetime.date(2019, 9, 1)])
    #    #a.set_xlim([datetime.date(2002,8,1), datetime.date(2003, 8, 1)])
    #    #a.set_xlim([datetime.date(2004,1,1), datetime.date(2004, 8, 1)])
    #    a.set_xlim([datetime.date(2002,10,1), datetime.date(2003, 4, 1)])

    props = dict(boxstyle='round', facecolor='white', alpha=0.0, ec="white")
    ax1.text(0.01, 0.96, "(a)", transform=ax1.transAxes, fontweight='bold',
             fontsize=12, verticalalignment='top', bbox=props)
    ax2.text(0.01, 0.96, "(b)", transform=ax2.transAxes, fontweight='bold',
             fontsize=12, verticalalignment='top', bbox=props)
    ax3.text(0.01, 0.96, "(c)", transform=ax3.transAxes, fontweight='bold',
             fontsize=12, verticalalignment='top', bbox=props)
    ax4.text(0.01, 0.96, "(d)", transform=ax4.transAxes, fontweight='bold',
             fontsize=12, verticalalignment='top', bbox=props)
    ax5.text(0.01, 0.96, "(e)", transform=ax5.transAxes, fontweight='bold',
             fontsize=12, verticalalignment='top', bbox=props)
    ax6.text(0.01, 0.96, "(f)", transform=ax6.transAxes, fontweight='bold',
             fontsize=12, verticalalignment='top', bbox=props)
    #"""
    if plot_fname is None:
        plt.show()
    else:
        fig.savefig(plot_fname, bbox_inches='tight', pad_inches=0.1)
    #"""
    #plt.show()

def read_cable_file(fname, type=None):

    if type == "CABLE":
        vars_to_keep = ['psi_leaf', 'gsw_sha', 'gsw_sun','weighted_psi_soil',\
                        'plc']
    elif type == "FLUX":
        vars_to_keep = ['GPP','Qle']
    elif type == "MET":
        vars_to_keep = ['Rainf']

    ds = xr.open_dataset(fname, decode_times=False)

    time_jump = int(ds.time[1].values) - int(ds.time[0].values)
    if time_jump == 3600:
        freq = "H"
    elif time_jump == 1800:
        freq = "30M"
    elif time_jump == 10800:
        freq = "3H"
    else:
        raise("Time problem")

    units, reference_date = ds.time.attrs['units'].split('since')
    #df = ds[vars_to_keep].squeeze(dim=["x","y","soil"], drop=True).to_dataframe()

    if type == "CABLE":
        ds = ds[vars_to_keep].squeeze(dim=["x","y","patch"], drop=True)

        """
        zse = np.array([.022, .058, .154, .409, 1.085, 2.872])

        frac1 = zse[0] / (zse[0] + zse[1])
        frac2 = zse[1] / (zse[0] + zse[1])
        frac3 = zse[2] / (zse[2] + zse[3])
        frac4 = zse[3] / (zse[2] + zse[3])
        frac5 = zse[4] / (zse[4] + zse[5])
        frac6 = zse[5] / (zse[4] + zse[5])

        ds['theta1'] = (ds['SoilMoist'][:,0] * frac1) + \
                       (ds['SoilMoist'][:,1] * frac2)
        ds['theta2'] = (ds['SoilMoist'][:,2] * frac3) + \
                       (ds['SoilMoist'][:,3] * frac4)
        ds['theta3'] = (ds['SoilMoist'][:,4] * frac5) + \
                       (ds['SoilMoist'][:,5] * frac6)
        """
        start = reference_date.strip().split(" ")[0].replace("-","/")
        ds['time'] = pd.date_range(start=start, periods=len(ds.psi_leaf), freq=freq)
    elif type == "MET":
        ds = ds[vars_to_keep].squeeze(dim=["x","y"], drop=True)

        start = reference_date.strip().split(" ")[0].replace("-","/")
        ds['time'] = pd.date_range(start=start, periods=len(ds.Rainf), freq=freq)
    elif type == "FLUX":
        ds = ds[vars_to_keep].squeeze(dim=["x","y"], drop=True)

        start = reference_date.strip().split(" ")[0].replace("-","/")
        ds['time'] = pd.date_range(start=start, periods=len(ds.GPP), freq=freq)

    #print(ds['time.month'])
    #print(ds['time.dayofyear'])
    #print(ds['time.dayofweek'])
    #print(ds['time.hour'])
    # print(ds.sel(time=datetime.time(12)))
    #print(ds['time.hour'])

    return ds



if __name__ == "__main__":


    fname1 = "outputs/hydraulics_profitmax_-30.40_151.60.nc"
    fname2 = "outputs/hydraulics_profitmax_-30.40_151.60_low_kcrit.nc"
    fname3 = "outputs/hydraulics_profitmax_-30.40_151.60_lower_kcrit.nc"
    fname4 = "outputs/hydraulics_profitmax_-30.40_151.60_half_kmax.nc"
    fname5 = "outputs/hydraulics_profitmax_-30.40_151.60_kmax_0.3.nc"
    fname6 = "outputs/hydraulics_profitmax_-30.40_151.60_higher_lai.nc"
    fname7 = "outputs/hydraulics_profitmax_-30.40_151.60_double_co2_higher_lai.nc"

    plot_fname = "/Users/mdekauwe/Dropbox/Documents/papers/Future_euc_drought_paper/figures/figs/sensitivity.pdf"
    #plot_fname = "/Users/mdekauwe/Desktop/sensitivity.pdf"

    main(fname1, fname2, fname3, fname4, fname5, fname6, fname7, plot_fname)
