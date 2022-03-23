#!/usr/bin/env python
"""
Plot the SPI for the 2017-2019 drought. Add a scalebar and map of Aus
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
import cartopy.crs as ccrs

#def scale_bar(ax, length=None, location=(0.88, 0.035), linewidth=3):
def scale_bar(ax, length=None, location=(0.68, 0.05), linewidth=3):
    """
    ax is the axes to draw the scalebar on.
    length is the length of the scalebar in km.
    location is center of the scalebar in axis coordinates.
    (ie. 0.5 is the middle of the plot)
    linewidth is the thickness of the scalebar.
    """
    #Get the limits of the axis in lat long
    llx0, llx1, lly0, lly1 = ax.get_extent(ccrs.PlateCarree())
    #Make tmc horizontally centred on the middle of the map,
    #vertically at scale bar location
    sbllx = (llx1 + llx0) / 2
    sblly = lly0 + (lly1 - lly0) * location[1]
    tmc = ccrs.TransverseMercator(sbllx, sblly)

    #Get the extent of the plotted area in coordinates in metres
    x0, x1, y0, y1 = ax.get_extent(tmc)
    #Turn the specified scalebar location into coordinates in metres
    sbx = x0 + (x1 - x0) * location[0]
    sby = y0 + (y1 - y0) * location[1]

    #Calculate a scale bar length if none has been given
    #(Theres probably a more pythonic way of rounding the number but this works)
    if not length:
        length = (x1 - x0) / 5000 #in km
        ndim = int(np.floor(np.log10(length))) #number of digits in number
        length = round(length, -ndim) #round to 1sf
        #Returns numbers starting with the list
        def scale_number(x):
            if str(x)[0] in ['1', '2', '5']: return int(x)
            else: return scale_number(x - 10 ** ndim)
        length = scale_number(length)

    #Generate the x coordinate for the ends of the scalebar
    bar_xs = [sbx - length * 500, sbx + length * 500]


    # offset the line isn't quite straight, presumably related to the projection
    # bit of a hack to roughly fix for visual purposes.
    offset = 12000
    #offset2 = 5000
    offset2 = 16000
    #Plot the scalebar
    #ax.plot(bar_xs, [sby, sby], transform=tmc, color='k', linewidth=linewidth)
    ax.plot(bar_xs, [sby-offset, sby-offset2], transform=tmc, color='k',
            linewidth=linewidth)
    print(bar_xs, sby, sby)
    print(tmc)

    #Plot the scalebar label
    ax.text(sbx, sby, str(length) + ' km', transform=tmc, fontsize=16,
            horizontalalignment='center', verticalalignment='bottom')

def main(spi, ofname, plot_dir):


    fig = plt.figure(figsize=(30, 10))
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.size'] = "16"
    plt.rcParams['font.sans-serif'] = "Helvetica"


    #cmap = plt.cm.get_cmap('YlOrRd', 9) # discrete colour map
    cmap = plt.cm.get_cmap('RdYlBu') # continuous colour map
    #cmap = plt.cm.get_cmap('viridis_r') # continuous colour map

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

    i=0
    #plims = plot_map(ax, data, cmap, i)
    plims = plot_map(axgr[0], spi, cmap, i)
    #plims = plot_map(ax, ds.plc[0,0,:,:], cmap, i)

    import cartopy.feature as cfeature
    states = cfeature.NaturalEarthFeature(category='cultural',
                                          name='admin_1_states_provinces_lines',
                                          scale='10m',facecolor='none')

    # plot state border
    SOURCE = 'Natural Earth'
    LICENSE = 'public domain'
    axgr[0].add_feature(states, edgecolor='black', lw=0.5)

    axgr[0].text(146, -32, 'New South Wales', horizontalalignment='center',
            transform=ccrs.PlateCarree(), fontsize=14)
    axgr[0].text(145, -36.8, 'Victoria', horizontalalignment='center',
            transform=ccrs.PlateCarree(), fontsize=14)

    axgr[0].plot(151.2093, -33.8688, 'ko', markersize=4, transform=ccrs.PlateCarree())
    axgr[0].text(151.1, -33.868, 'Sydney', horizontalalignment='right',
            transform=ccrs.PlateCarree(), fontsize=10)


    axgr[0].plot(149.1300, -35.2809, 'ko', markersize=4, transform=ccrs.PlateCarree())
    axgr[0].text(148.900, -35.0809, 'Canberra', horizontalalignment='center',
            transform=ccrs.PlateCarree(), fontsize=10)

    axgr[0].plot(144.9631, -37.8136, 'ko', markersize=4, transform=ccrs.PlateCarree())
    axgr[0].text(144.85, -37.8136, 'Melbourne', horizontalalignment='right',
            transform=ccrs.PlateCarree(), fontsize=10)

    gl = axgr[0].gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=0.5, color='black', alpha=0.5,
                      linestyle='--')


    gl.left_labels = True
    gl.bottom_labels = True
    #"""
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xlines = False
    gl.ylines = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    gl.xlocator = mticker.FixedLocator([141, 145,  149, 153])
    gl.ylocator = mticker.FixedLocator([-29, -32, -35, -38])

    scale_bar(axgr[0], 100)


    from geemap import cartoee
    #cartoee.add_scale_bar_lite(axgr[0], 100, xy=(0.9, 0.02), linewidth=3, fontsize=16, color="black", unit="km")
    #cartoee.add_north_arrow(axgr[0], 'N', xy=(0.95, 0.19), arrow_length=0.1,
    #                        fontsize=16, text_color="black", arrow_color="black")
    cbar = axgr.cbar_axes[0].colorbar(plims)
    cbar.ax.set_title("SPI (-)", fontsize=16, pad=10)
    cbar.ax.tick_params(labelsize=16)

    import cartopy
    import cartopy.mpl.geoaxes
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    axins = inset_axes(axgr[0], width="20%", height="20%", loc="lower right",
                       axes_class=cartopy.mpl.geoaxes.GeoAxes,
                       axes_kwargs=dict(map_projection=cartopy.crs.PlateCarree()))
    axins.add_feature(cartopy.feature.COASTLINE, edgecolor='black', lw=0.5)
    axins.set_extent([112, 155, -44, -9], ccrs.PlateCarree())
    axins.coastlines(resolution='110m')

    axins.add_feature(states, edgecolor='black', lw=0.5)

    import shapely.geometry as sgeom
    box = sgeom.box(minx=140, maxx=154, miny=-39.2, maxy=-28)
    x0, y0, x1, y1 = box.bounds

    proj = ccrs.PlateCarree()
    box_proj = ccrs.PlateCarree()


    axins.add_geometries([box], box_proj, facecolor='coral',
                         edgecolor='black', alpha=0.5)





    ofname = os.path.join(plot_dir, ofname)
    fig.savefig(ofname, dpi=300, bbox_inches='tight',
                pad_inches=0.1)



def plot_map(ax, var, cmap, i):
    print(np.nanmin(var), np.nanmax(var))
    #vmin, vmax = 0, 90
    vmin, vmax = -1.5, 1.5
    #top, bottom = 90, -90
    #left, right = -180, 180
    top, bottom = -10, -44.5
    left, right = 112, 156.2
    img = ax.imshow(var, origin='upper',
                    transform=ccrs.PlateCarree(),
                    interpolation=None, cmap=cmap,
                    extent=(left, right, bottom, top),
                    vmin=vmin, vmax=vmax)
    ax.coastlines(resolution='10m', linewidth=1.0, color='black')
    #ax.add_feature(cartopy.feature.OCEAN)

    ax.set_xlim(140, 154)
    ax.set_ylim(-39.2, -28)

    return img


if __name__ == "__main__":

    plot_dir = "/Users/mdekauwe/Dropbox/Documents/papers/Future_euc_drought_paper/figures/figs/"
    #plot_dir = "/Users/mdekauwe/Desktop/"
    fname = "Monthly_SPI_values_scale_6_AWAP_1911_2016.nc"
    ds = xr.open_dataset(fname, decode_times=False)

    # We want the mean anomaly Jan 2017 to Aug 2019
    st = 1308-36 # Jan 2017
    en = 1308-4  # Aug 2019
    spi = ds.SPI[st:en,:,:].mean(axis=0)

    #print(ds.longitude[0], ds.longitude[-1])
    #print(ds.latitude[0], ds.latitude[-1])

    ofname = "SPI_2017_2019_anomaly.png"
    main(spi, ofname, plot_dir)
