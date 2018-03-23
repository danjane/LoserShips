import numpy as np
import pandas as pd
import scipy.spatial
import datetime

#

def so2dispersion_distance2_simple(b_vec, s_vec):
    if s_vec[2] < b_vec[2]:
        # satellite took reading before boat was there
        return 1267650600228229401496703205376 #very far away
    else:
        # use rough calc of a metre
        lat_adjustment = 111412.84 * np.cos(np.deg2rad(s_vec[1]))
        lon_adjustment = 111132.92
        time_adjustment = 24 * 1000 #one hour about a km

        diff_vec = b_vec-s_vec

        return np.dot(np.dot(diff_vec, np.diag([lat_adjustment, lon_adjustment, time_adjustment])), diff_vec)


def boat_satellite_distance(bs, ss):
    ds = scipy.spatial.distance.cdist(bs[:,:3], ss[:,:3], so2dispersion_distance2_simple)
    return np.sqrt(ds)


def convert_satellite_pandas(s_pd):
    pos = s_pd[['Latitude', 'Longitude']].values
    timestamp2matlabdn = lambda x: (datetime.datetime.strptime(x, '%Y-%m-%d') + datetime.timedelta(days=366)).toordinal()
    time = s_pd['Date'].apply(timestamp2matlabdn)
    time = time.values + s_pd['Time'].values/24/60/60

    return np.concatenate([pos, time[:, None]], axis=1)