# NB: I probably shouldn't do this, but I'm storing the times in
# Matlab format so I can easily do linear algebra later.
#
# Work in Lon-Lat (x,y)

import numpy as np
import pandas as pd
import scipy.spatial
import datetime
import pyproj
import matplotlib.pyplot as plt


def lat_adj2(lat_degrees):
    return EQUATOR_LAT_ADJ2 * np.cos(np.deg2rad(lat_degrees))**2


def distance2_simple(b_lonlat, s_lonlat):
        # use rough calc of a metre
        lat_adjustment2 = lat_adj2(s_lonlat[1])
        diff_vec = b_lonlat-s_lonlat
        return np.dot(np.dot(diff_vec, np.diag([lat_adjustment2, LON_ADJ2])), diff_vec.T)


def so2dispersion_distance2_simple(b_vec, s_vec):
    if s_vec[2] < b_vec[2]:
        # satellite took reading before boat was there
        return 1267650600228229401496703205376 #very far away
    else:
        # use rough calc of a metre
        lat_adjustment2 = lat_adj2(s_vec[1])
        diff_vec = b_vec-s_vec
        return np.dot(np.dot(diff_vec, np.diag([lat_adjustment2, LON_ADJ2, TIME_ADJ2])), diff_vec.T)


def boat_satellite_distance(bs, ss):
    ds = scipy.spatial.distance.cdist(bs[:, :3], ss[:, :3], so2dispersion_distance2_simple)
    return np.sqrt(ds)


def timestamp2matlabdn(date_str):
    date_num = datetime.datetime.strptime(date_str, '%Y-%m-%d') + datetime.timedelta(days=366)
    return date_num.toordinal()


def convert_satellite_pandas(s_pd):
    pos = s_pd[['Longitude', 'Latitude']].values

    time = s_pd['Date'].apply(timestamp2matlabdn)
    time = time.values + s_pd['Time'].values/24/60/60

    return np.concatenate([pos, time[:, None]], axis=1)


def investigate_interesting_boat_position(b_vec, s_pd, idx_closest):
    # Ok. We've found a satellite reading pretty close to the boat so now we want to see whether the so2 reading
    # was within the null hypothesis.
    # Note - this is irrespective of whether the boat is polluting, just did we detect anything at all

    s_closest = s_pd.iloc[idx_closest]

    # (Maybe I'll use az with the wind one day...)
    (az12, az21, dist) = wgs84_geod.inv(b_vec[0], b_vec[1], s_closest['Longitude'], s_closest['Latitude'])

    # Get all satellite points within about 2000km of the boat
    ds2 = scipy.spatial.distance.cdist([b_vec[:2]], s_pd[['Longitude','Latitude']].values, distance2_simple)
    idx = ds2 < (2000 * 1000)**2

    # so2 detected distribution
    so2 = s_pd[idx.T]['ColumnAmountSO2_PBL'].values

    return s_closest, so2, idx


def plot_histogram(s_closest, so2):
    n, bins, patches = plt.hist(so2, 50, facecolor='green', alpha=0.75)
    plt.show()


def find_min_idx(x):
    k = x.argmin()
    ncol = x.shape[1]
    return k // ncol, k % ncol


wgs84_geod = pyproj.Geod(ellps='WGS84')
EQUATOR_LAT_ADJ2 = 111412.84**2
LON_ADJ2 = 111132.92**2
TIME_ADJ2 = (24 * 1000)**2 #one hour about a km
