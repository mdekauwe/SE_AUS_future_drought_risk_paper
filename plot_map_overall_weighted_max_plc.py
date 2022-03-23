#!/usr/bin/env python
"""
Equally weight the PLC for each pixel by species occurrence. We are just doing
the CTL experiment

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

    # calcualted weighted plc
    overall = np.zeros((681,841))
    count = np.zeros((681,841))

    for i in range(len(species_list)):

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
        rplc = data / 88.


        overall = np.where(~np.isnan(rplc), overall+rplc, overall)
        count = np.where(~np.isnan(rplc), count+1.0, count)


    count = np.where(count==0.0, np.nan, count)

    overall /= count

    fig = plt.figure(figsize=(30, 10))
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.size'] = "14"
    plt.rcParams['font.sans-serif'] = "Helvetica"


    #cmap = plt.cm.get_cmap('YlOrRd', 9) # discrete colour map
    cmap = plt.cm.get_cmap('YlOrRd') # continuous colour map

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=projection))
    rows = 1
    cols = 1

    axgr = AxesGrid(fig, 111, axes_class=axes_class,
                    nrows_ncols=(rows, cols),
                    axes_pad=0.35,
                    cbar_location='right',
                    cbar_mode='single',
                    cbar_pad=0.2,
                    cbar_size='5%',
                    label_mode='')  # note the empty label_mode


    #plims = plot_map(ax, data, cmap, i)
    plims = plot_map(axgr[0], overall, cmap, i)
    #plims = plot_map(ax, ds.plc[0,0,:,:], cmap, i)


    cbar = axgr.cbar_axes[0].colorbar(plims)
    cbar.ax.set_title("PLC$_{\mathrm{max}}$ (-)", fontsize=14, pad=10)
    ofname = os.path.join(plot_dir, ofname)
    fig.savefig(ofname, dpi=300, bbox_inches='tight',
                pad_inches=0.1)



def plot_map(ax, var, cmap, i):
    #print(np.nanmin(var), np.nanmax(var))
    #vmin, vmax = 0, 90
    vmin, vmax = 0, 1
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


    gl.left_labels = True
    gl.bottom_labels = True
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


    fname = "plc_max_all_experiments.csv"
    df = pd.read_csv(fname)
    df = df[df.plc_max > -500.].reset_index()


    ofname = "PLC_max_overall_weighted.png"
    main(df, "control", species_list, nice_species_list, ofname, plot_dir)

    """
    ofname = "PLC_max_ePPT_drought.png"
    main(df, "ePPT", species_list, nice_species_list, ofname, plot_dir)

    ofname = "PLC_max_eCO2_ePPT_drought.png"
    main(df, "eCO2_ePPT", species_list, nice_species_list, ofname, plot_dir)
    """
