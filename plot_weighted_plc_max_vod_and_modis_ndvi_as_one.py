#!/usr/bin/env python
"""
Using three panels, plot the weighted PLC by species occurrence, the VOD anomaly
and the NDVI anomaly. Only for the CTL experiment

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
import xarray as xr

def main(df, experiment, species_list, nice_species_list, ofname, plot_dir,
         fname2, fname3):

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
        #rplc = data / 88.
        rplc = data


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
    cols = 3

    axgr = AxesGrid(fig, 121, axes_class=axes_class,
                    nrows_ncols=(rows, cols),
                    axes_pad=0.6,
                    cbar_location='right',
                    cbar_mode='each',
                    cbar_pad=0.1,
                    cbar_size='5%',
                    label_mode='')  # note the empty label_mode


    plims = plot_map(axgr[0], overall, cmap, i, grid_left=True, top=-10,
                     bottom=-44, left=112, right=154, vmin=0,
                     vmax=88, origin='lower')

    """
    # Write to netcdf file for stats...we need to degrade and match to coarse
    # data in cdo
    nrows = 681
    ncols = 841
    resolution = 0.05
    #lats = np.zeros((nrows, ncols))
    #lons = np.zeros((nrows, ncols))
    lats = np.zeros(nrows)
    lons = np.zeros(ncols)

    # Upper right corner.
    urcrnrlat = -44. + (float(nrows) * resolution)
    urcrnrlon = 112. + (float(ncols) * resolution)

    # lower left corner
    llcrnrlon = 112.
    llcrnrlat = -44.

    #for ii in range(nrows):
    #    for jj in range(ncols):
    #        lats[ii,jj] = urcrnrlat - (float(ii) * resolution)
    #        lons[ii,jj] = llcrnrlon + (float(jj) * resolution)

    for ii in range(nrows):
        lats[ii] = urcrnrlat - (float(ii) * resolution)
    for jj in range(ncols):
        lons[jj] = llcrnrlon + (float(jj) * resolution)

    ds = xr.DataArray(np.flipud(overall),
                      coords=[('lat', lats), ('lon', lons)], name='plc_weight')
    ds.to_netcdf('/Users/mdekauwe/Desktop/weighted_plc.nc')
    """


    cbar = axgr.cbar_axes[0].colorbar(plims)
    cbar.ax.tick_params(labelsize=14)
    cbar.ax.set_title("PLC (%)", fontsize=14, pad=10)


    ds = xr.open_dataset(fname2, decode_times=False)
    data = ds.layer.values
    lats = ds.latitude.values
    lons = ds.longitude.values
    ncols = len(np.unique(lons))
    nrows = len(np.unique(lons))
    top = lats[0]
    bottom = lats[-1]
    left = lons[0]
    right = lons[-1]


    #cmap2 = plt.cm.get_cmap('BrBG', 12) # discrete colour map
    cmap2 = plt.cm.get_cmap('BrBG') # discrete colour map

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=projection))

    plims2 = plot_map(axgr[1], data, cmap2, i, grid_left=False, top=top,
                     bottom=bottom, left=left,
                     right=right, vmin=-30, vmax=30, origin='upper')


    ds = xr.open_dataset(fname3)
    bottom, top = np.min(ds.lat).values, np.max(ds.lat).values
    left, right = np.min(ds.lon).values, np.max(ds.lon).values

    chg = ds.Band1[:,:].values
    #cmap3 = plt.cm.get_cmap('BrBG', 12) # discrete colour map
    cmap3 = plt.cm.get_cmap('BrBG') # discrete colour map

    projection = ccrs.PlateCarree()
    axes_class = (GeoAxes, dict(map_projection=projection))

    plims3 = plot_map(axgr[2], chg, cmap3, i, grid_left=False, top=top,
                     bottom=bottom, left=left,
                     right=right, vmin=-40, vmax=40, origin='lower')


    for i, ax in enumerate(axgr):
        import cartopy.feature as cfeature
        states = cfeature.NaturalEarthFeature(category='cultural',
                                              name='admin_1_states_provinces_lines',
                                              scale='10m',facecolor='none')

        # plot state border
        SOURCE = 'Natural Earth'
        LICENSE = 'public domain'
        ax.add_feature(states, edgecolor='black', lw=0.5)

    cbar2 = axgr.cbar_axes[1].colorbar(plims2)
    cbar2.ax.set_title("% Difference", fontsize=14, pad=10)
    cbar3 = axgr.cbar_axes[2].colorbar(plims3)
    cbar3.ax.set_title("% Difference", fontsize=14, pad=10)


    props = dict(boxstyle='round', facecolor='white', alpha=0.0, ec="white")
    ax.text(0.9, 0.09, "(a)", transform=axgr[0].transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    props = dict(boxstyle='round', facecolor='white', alpha=0.0, ec="white")
    ax.text(0.9, 0.09, "(b)", transform=axgr[1].transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    props = dict(boxstyle='round', facecolor='white', alpha=0.0, ec="white")
    ax.text(0.9, 0.09, "(c)", transform=axgr[2].transAxes, fontsize=12,
             verticalalignment='top', bbox=props)

    ofname = os.path.join(plot_dir, ofname)
    fig.savefig(ofname, dpi=300, bbox_inches='tight',
                pad_inches=0.1)


def plot_map(ax, var, cmap, i, grid_left=True, top=None, bottom=None, left=None,
             right=None, vmin=None, vmax=None, origin=None):
    #print(np.nanmin(var), np.nanmax(var))
    #vmin, vmax = 0, 90

    img = ax.imshow(var, origin=origin,
                    transform=ccrs.PlateCarree(),
                    interpolation=None, cmap=cmap,
                    extent=(left, right, bottom, top),
                    vmin=vmin, vmax=vmax)
    ax.coastlines(resolution='10m', linewidth=1.0, color='black')
    #ax.add_feature(cartopy.feature.OCEAN)

    #ax.set_xlim(140, 154)
    #ax.set_ylim(-39.4, -28)

    ax.set_xlim(140.7, 154)
    ax.set_ylim(-39.2, -28.1)

    #"""
    if i == 0 or i == 5 or i > 9:

        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=0.5, color='black', alpha=0.5,
                          linestyle='--')
    else:
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                          linewidth=0.5, color='black', alpha=0.5,
                          linestyle='--')

    if grid_left:
        gl.left_labels = True
    else:
        gl.left_labels = False
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

def calc_average(ds, start_year, end_year, end_month, stat):

    ndates, nrows, ncols = ds.NDVI.shape
    nyears = (end_year - start_year) + 1
    nseas = 4
    ndvi = np.zeros((nyears*nseas,nrows,ncols))

    if end_year == 2019:
        ndvi = np.zeros(((nyears*nseas)-1,nrows,ncols))
    count = 0
    for i in range(ndates):
        date = ds.time.values[i]
        year = int(str(ds.time.values[i]).split("-")[0])
        month = str(ds.time.values[i]).split("-")[1]

        if year >= start_year and year <= end_year:

            if year == 2019 and month == "10":
                print ("bad")
            else:
                #print(date)
                ndvi[count,:,:] = np.where(~np.isnan(ds.NDVI[i,:,:]), \
                                        ds.NDVI[i,:,:], ndvi[count,:,:])
                count += 1

    if stat == "mean":
        ndvi = np.nanmean(ndvi, axis=0)
    else:
        ndvi = np.nanmin(ndvi, axis=0)
    ndvi = np.where(ndvi <= 0.05, np.nan, ndvi)

    return (ndvi)


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


    fname = "plc_max_all_experiments.csv"
    df = pd.read_csv(fname)
    df = df[df.plc_max > -500.].reset_index()

    fname2 = "/Users/mdekauwe/research/VOD_anomaly_2017_2019/vod_mean_diff_ref2002-2016_bigdry2017-2019_v3.nc"
    fname3 = "/Users/mdekauwe/research/SE_AUS_drought_risk_paper/NDVI_MODIS/MOD13A2_NDVI_0p025_2017-01_2019-8_annual-min-relative-to-NDVI_MA.nc"
    ofname = "PLC_max_overall_weighted_and_vod_ndvi_map.png"
    main(df, "control", species_list, nice_species_list, ofname, plot_dir, fname2, fname3)
