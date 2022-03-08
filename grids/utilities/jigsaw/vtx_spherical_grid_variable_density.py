#!/usr/bin/env python
#
#  Basic script to creat spherical grids for MPAS-Atmosphere
#  Based on 
#  http://mpas-dev.github.io/MPAS-Tools/stable/mesh_creation.html#spherical-meshes
#
#   Install the conda enviroment MPAS-Tools
# 1) Get the requirements https://github.com/pedrospeixoto/MPAS-Tools/blob/master/conda_package/dev-spec.txt
# 2) Create enviroment
#     $ conda config --add channels conda-forge
#     $ conda create --name mpas-tools --file dev-spec.txt 
# 3) Install the mpas-tools pack
#     $ conda install mpas_tools
# 4) Use it with $ conda activate mpas-tools
#
 
import numpy as np
from mpas_tools.ocean import build_spherical_mesh
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import os


def cellWidthVsLatLon():
    """
    Create cell width array for this mesh on a regular latitude-longitude grid.
    Returns
    -------
    cellWidth : ndarray
        m x n array of cell width in km
    lon : ndarray
        longitude in degrees (length n and between -180 and 180)
    lat : ndarray
        longitude in degrees (length m and between -90 and 90)
    """
    dlat = 1
    dlon = 1
    constantCellWidth = 24

    nlat = int(180/dlat) + 1
    nlon = int(360/dlon) + 1

    lat = np.linspace(-90., 90., nlat)
    lon = np.linspace(-180., 180., nlon)

    cellWidth = constantCellWidth * np.ones((lat.size, lon.size))
    return cellWidth, lon, lat


def latlon_to_distance_center(lon, lat):
    lon, lat = map(np.radians, [lon, lat])

    haver_formula = np.sin(lat / 2.0) ** 2 + \
                    np.cos(lat) * np.sin(lon / 2.0) ** 2

    dists = 2 * np.arcsin(np.sqrt(haver_formula)) * 6367
    return dists


def localrefVsLatLon(r, maxdist, slope, gammas=500, maxepsilons=10000):
    """
    Create cell width array for this mesh on a locally refined latitude-longitude grid.
    Input
    ---------
    r : float
        minimun desired cell width resolution in km

    Returns
    -------
    cellWidth : ndarray
        m x n array of cell width in km
    lon : ndarray
        longitude in degrees (length n and between -180 and 180)
    lat : ndarray
        longitude in degrees (length m and between -90 and 90)

    # Radius (in km) of high resolution area
    maxdist = 15*r
    # (increase_of_resolution) / (distance)
    slope = 10. / 50.
    # Gammas
    gammas = 500.
    # distance (in km) of transition zone belt: ratio / slope
    maxepsilons = 10000.
    """
    constantCellWidth = r  # in km

    dlat = constantCellWidth / 110.
    dlon = dlat

    nlat = int(180. / dlat) + 1
    nlon = int(360. / dlon) + 1

    lat = np.linspace(-90., 90., nlat)
    lon = np.linspace(-180., 180., nlon)

    lons, lats = np.meshgrid(lon, lat)
    dists = latlon_to_distance_center(lons, lats)

    # Parameters
    # ------------------------------
    epsilons = gammas / slope

    if epsilons > maxepsilons:
        epsilons = maxepsilons

    # initialize with resolution = r (min resolution)
    resolution = r * np.ones(np.shape(dists))

    # point in transition zone
    transition_zone = (dists > maxdist) & (dists <= maxdist + epsilons)
    sx = (dists - maxdist) * slope
    transition_values = r + sx
    resolution = np.where(transition_zone, transition_values, resolution)

    # further points
    far_from_center = (dists > maxdist + epsilons)
    resolution[far_from_center] += epsilons * slope

    return resolution, lon, lat


def sectionsrefVsLatLon(r, maxdist, maxr=15., epsilons=40.,
                        width_mid_stop=40.):
    """
    Create cell width array for this mesh on a locally refined latitude-longitude grid.
    Input
    ---------
    r : float
        minimun desired cell width resolution in km

    Returns
    -------
    cellWidth : ndarray
        m x n array of cell width in km
    lon : ndarray
        longitude in degrees (length n and between -180 and 180)
    lat : ndarray
        longitude in degrees (length m and between -90 and 90)

    """
    constantCellWidth = r  # in km

    dlat = constantCellWidth / 110.
    dlon = dlat

    nlat = int(180. / dlat) + 1
    nlon = int(360. / dlon) + 1

    lat = np.linspace(-90., 90., nlat)
    lon = np.linspace(-180., 180., nlon)

    lons, lats = np.meshgrid(lon, lat)
    dists = latlon_to_distance_center(lons, lats)

    # Parameters
    # ------------------------------
    slope = (maxr - r) / epsilons
    final_res_dist = 1000

    # initialize with resolution = r (min resolution)
    resolution = r * np.ones(np.shape(dists))

    # point in transition zone
    transition_zone = (dists > maxdist) & (dists <= maxdist + epsilons)
    sx = (dists - maxdist) * slope
    transition_values = r + sx
    resolution = np.where(transition_zone, transition_values, resolution)
    final_res1 = r + epsilons * slope

    transition_zone2 = (dists > maxdist + epsilons) & \
                       (dists <= maxdist + epsilons + width_mid_stop)
    transition_values2 = final_res1
    resolution = np.where(transition_zone2, transition_values2, resolution)

    transition_zone3 = (dists > maxdist + epsilons + width_mid_stop) & \
                       (dists <= final_res_dist)
    sx = (dists - (maxdist + epsilons + width_mid_stop)) * slope
    transition_values3 = final_res1 + sx
    resolution = np.where(transition_zone3, transition_values3, resolution)
    final_res3 = final_res1 + (final_res_dist - (maxdist + epsilons +
                                                 width_mid_stop)) * slope

    # further points
    far_from_center = (dists > final_res_dist)
    resolution[far_from_center] = final_res3

    return resolution, lon, lat


def viewcelWidth(cellWidth, lat, lon, name):

    kwargs = {'cmap': 'Spectral', 'vmin': 0, 'vmax': 20, 'levels': 21}

    print('View')
    lons, lats = np.meshgrid(lon, lat)
    dists = latlon_to_distance_center(lons, lats)

    ds = xr.Dataset({'resolution': (('lat', 'lon'), cellWidth),
                     'distance': (('lat', 'lon'), dists)},
                    coords={'lat': lat, 'lon': lon})

    axis = ds.interp(lat=0, method='nearest').sel(lon=slice(0, 180))
    print(axis)
    ds2 = xr.Dataset({'resolution': ('distance', axis['resolution'].values)},
                     coords={'distance': axis['distance'].values})

    ds2['resolution'].plot()
    plt.savefig(name + '_bydistance.png')
    plt.close()

    ds2['resolution'].where(ds2['distance'] < 500).plot()
    plt.savefig(name + '_bydistanceless500.png')
    plt.close()

    ds2['resolution'].where(ds2['distance'] < 100).plot()
    plt.savefig(name + '_bydistanceless100.png')
    plt.close()

    # REGION
    region = ds.sel(lat=slice(-1, 1), lon=slice(-1, 1))
    print(region)

    for dist in [500, 100, 50]:
        region['resolution'].where(region['distance'] < dist).plot(**kwargs)
        plt.title('Distance closer than ' + str(dist) + 'km')
        plt.savefig(name + '_less' + str(dist) + 'km.png')
        plt.close()

    for res in [20, 10, 4]:
        region['resolution'].where(region['resolution'] < res).plot(**kwargs)
        plt.title('Resolution lower than ' + str(res) + 'km')
        plt.savefig(name + '_res' + str(res) + 'km.png')
        plt.close()


def main():
    name = 'light2'

    os.system('mkdir -p ' + name)

    if 'jigsaw' in name:
        r = 1.
        md = 10*r
        slope = 1. / 7
        cellWidth, lon, lat = localrefVsLatLon(r, maxdist=md, slope=slope,
                                               gammas=500, maxepsilons=10000)
    else:
        radi = 50

        r = 1.
        maxr = 12

        md = 12
        epsilons = radi - md + maxr*3
        wms = maxr*6

        cellWidth, lon, lat = sectionsrefVsLatLon(r, maxdist=md,
                                                  maxr=maxr,
                                                  epsilons=epsilons,
                                                  width_mid_stop=wms)

    viewcelWidth(cellWidth, lat, lon, name + '/' + name)

    build_spherical_mesh(cellWidth, lon, lat,
                         out_filename=name + '/' + name + '.nc')

    os.system('./region.sh ' + name + '/' + name + '.nc')


if __name__ == '__main__':
    main()