#!/usr/bin/env python
"""
Plot each species maximum PLC as map for each of the three experiments
(CTL, rPPT and eCO2) - halving kmax

That's all folks.
"""
__author__ = "Martin De Kauwe"
__version__ = "1.0 (23.03.2022)"
__email__ = "mdekauwe@gmail.com"

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import cartopy.crs as ccrs
import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import sys
import matplotlib.ticker as mticker
from cartopy.mpl.geoaxes import GeoAxes
from mpl_toolkits.axes_grid1 import AxesGrid
import pandas as pd

def main(df, experiment, species_list, nice_species_list, ofname, plot_dir):

    df_map = pd.read_csv("euc_map_east.csv")
    df_map = df_map.sort_values(by=['map'])

    sorted_map = df_map['map'].values

    fig = plt.figure(figsize=(30, 10))
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.size'] = "16"
    plt.rcParams['font.sans-serif'] = "Helvetica"


    #cmap = plt.cm.get_cmap('YlOrRd', 9) # discrete colour map
    cmap = plt.cm.get_cmap('YlOrRd') # continuous colour map

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=projection))
    rows = 3
    cols = 5

    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(rows, cols),
                    axes_pad=0.35,
                    cbar_location='right',
                    cbar_mode='single',
                    cbar_pad=0.2,
                    cbar_size='5%',
                    label_mode='')  # note the empty label_mode



    for i, ax in enumerate(axgr):

        species_mtch = df[(df.species == species_list[i]) & \
                          (df.experiment == experiment)].reset_index()

        data = np.ones((681,841)) * np.nan
        for k in range(len(species_mtch)):
            data[int(species_mtch["row"][k]),\
                 int(species_mtch["col"][k])] = species_mtch["plc_max"][k]

        # Species location sampling if off Rach's 10km grid, so we have
        # a gap between pixels, interpolate for visual purposes

        # First loop over cols
        for jj in range(841):
            for ii in range(681):
                if ~np.isnan(data[ii, jj]):
                    if np.isnan(data[ii, jj+1]):
                        data[ii, jj+1] = (data[ii, jj] + data[ii, jj+2]) / 2.

        # Next loop over rows
        for ii in range(681):
            for jj in range(841):
                if ~np.isnan(data[ii, jj]):
                    if np.isnan(data[ii+1, jj]):
                        data[ii+1, jj] = (data[ii, jj] + data[ii+2, jj]) / 2.

        # add a subplot into the array of plots
        #ax = fig.add_subplot(rows, cols, i+1, projection=ccrs.PlateCarree())
        #rplc = data / 88.
        rplc = data
        #plims = plot_map(ax, data, cmap, i)
        plims = plot_map(ax, rplc, cmap, i)
        #plims = plot_map(ax, ds.plc[0,0,:,:], cmap, i)

        ax.set_title('$\it{%s}$' % (nice_species_list[i]), fontsize=18)
        ax.text(0.78, 0.045, "     MAP:\n%d mm yr$^{-1}$" % (sorted_map[i]),
                horizontalalignment='center', size=13, color="black",
                transform=ax.transAxes)



    cbar = axgr.cbar_axes[0].colorbar(plims)
    cbar.ax.tick_params(labelsize=16)
    #cbar.ax.set_title("PLC$_{\mathrm{max}}$ (%)", fontsize=12, pad=10)
    #cbar.ax.set_title("PLC$_{\mathrm{max}}$ (-)", fontsize=12, pad=10)
    cbar.ax.set_title("PLC (%)", fontsize=16, pad=10)
    ofname = os.path.join(plot_dir, ofname)
    fig.savefig(ofname, dpi=300, bbox_inches='tight',
                pad_inches=0.1)

    plt.show()

def plot_map(ax, var, cmap, i):
    #print(np.nanmin(var), np.nanmax(var))
    vmin, vmax = 0, 90
    #vmin, vmax = 0, 1
    #top, bottom = 90, -90
    #left, right = -180, 180
    top, bottom = -10, -44
    left, right = 112, 154
    img = ax.imshow(var, origin='lower',
                    transform=ccrs.PlateCarree(),
                    interpolation=None, cmap=cmap,
                    extent=(left, right, bottom, top),
                    vmin=vmin, vmax=vmax)
    ax.coastlines(resolution='10m', linewidth=1.0, color='black')
    #ax.add_feature(cartopy.feature.OCEAN)

    ax.set_xlim(140, 154)
    ax.set_ylim(-39.4, -28)

    #"""
    if i == 0 or i == 5 or i > 9:

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='black', alpha=0.5,
                          linestyle='--')
    else:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=0.5, color='black', alpha=0.5,
                          linestyle='--')

    if i < 9:
        gl.bottom_labels = False
    if i == 0 or i == 5 or i == 10:
        gl.left_labels = True
    else:
        gl.left_labels = False

    #"""
    gl.top_labels = False
    gl.right_labels = False
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.xlocator = mticker.FixedLocator([141, 145,  149, 153])
    gl.ylocator = mticker.FixedLocator([-29, -32, -35, -38])

    return img


if __name__ == "__main__":

    #plot_dir = "plots"
    #if not os.path.exists(plot_dir):
    #    os.makedirs(plot_dir)
    plot_dir = "/Users/mdekauwe/Dropbox/Documents/papers/Future_euc_drought_paper/figures/figs/"
    #plot_dir = "/Users/mdekauwe/Desktop/"
    # Orderd by MAP low to high
    species_list = ['Eucalyptus largiflorens','Eucalyptus camaldulensis',\
                    'Eucalyptus populnea','Eucalyptus sideroxylon',\
                    'Eucalyptus melliodora','Eucalyptus blakelyi',\
                    'Eucalyptus macrorhyncha','Eucalyptus viminalis',\
                    'Eucalyptus crebra','Eucalyptus obliqua',
                    'Eucalyptus globulus','Eucalyptus tereticornis',\
                    'Eucalyptus saligna','Eucalyptus dunnii',\
                    'Eucalyptus grandis']
    nice_species_list = [i.replace("_", " ").replace("Eucalyptus", "E.") \
                         for i in species_list]


    fname = "plc_max_all_experiments_halfkmax.csv"
    df = pd.read_csv(fname)
    df = df[df.plc_max > -500.].reset_index()


    ofname = "PLC_max_control_drought_halfkmax.png"
    main(df, "control", species_list, nice_species_list, ofname, plot_dir)

    ofname = "PLC_max_ePPT_drought_halfkmax.png"
    main(df, "ePPT", species_list, nice_species_list, ofname, plot_dir)

    ofname = "PLC_max_eCO2_ePPT_drought_halfkmax.png"
    main(df, "eCO2_ePPT", species_list, nice_species_list, ofname, plot_dir)
