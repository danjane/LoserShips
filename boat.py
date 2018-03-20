# Given two consecutive AIS readings from a boat (at A and B),
# I would like to know how the boat got from A to B, and at
# what times it was passing through these intermediate points.
# For now I will assume a straight line on the Earth with constant
# speed, but hopefully we can improve on this with COG data, etc.
#
# We will want to use a bit of linear algebra to search for
# relevant instances which would be detectable by a satellite, so
# I'll store the results in an np.array along with the velocity.
#
# NB: I probably shouldn't do this, but I'm storing the times in
# Matlab format so I can easily do linear algebra later.

import pyproj
import datetime
import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import folium

def datetime2matlabdn(dt):
    # https://stackoverflow.com/questions/8776414/python-datetime-to-matlab-datenum
    mdn = dt + datetime.timedelta(days=366)
    frac_seconds = (dt-datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0)).seconds / (24.0 * 60.0 * 60.0)
    frac_microseconds = dt.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
    return mdn.toordinal() + frac_seconds + frac_microseconds


def interpolate_positions_times_velocities(ptv_matrix):
    interpolated_ptvs = []
    for idx in range(ptv_matrix.shape[0]-1):
        # Loop down pairs doing a linear interpolation
        startlat, startlong, starttime = ptv_matrix[idx, :3]
        endlat, endlong, endtime = ptv_matrix[idx+1, :3]

        interpolated_ptvs.append(
            interpolate_linear_position(startlat, startlong, starttime, endlat, endlong, endtime))

        #print("{}".format(interpolated_ptvs[-1].shape))

    #pickle.dump(interpolated_ptvs, open('ina.pickle', "wb"))
    return np.concatenate(interpolated_ptvs, axis=0)


def interpolate_linear_position(startlat, startlong, starttime, endlat, endlong, endtime):
    # Intermediate locations
    g = pyproj.Geod(ellps='WGS84')
    (az12, az21, dist) = g.inv(startlong, startlat, endlong, endlat)

    dist /= 1000.  # switch to km
    # calculate line string along path with segments <= 1 km
    lonlats = g.npts(startlong, startlat, endlong, endlat,
                     1 + int(dist))

    # npts doesn't include start/end points, so prepend/append them
    lonlats.insert(0, (startlong, startlat))
    lonlats.append((endlong, endlat))
    lonlats = np.asarray(lonlats)  # warning! still y-x (=lon,lat)

    # Intermediate times
    n_points = lonlats.shape[0]
    times = np.linspace(starttime, endtime, n_points)
    # d_time = endtime - starttime
    # i_time = d_time / (n_points - 1)
    # times = np.zeros((n_points, 1))
    # for i in range(n_points):
    #     times[i] = datetime2matlabdn(starttime + i_time * i)

    # Intermediate velocities
    speed = dist / (times[-1] - times[0]) / 24.  # currently assuming constant speed
    velocities = np.zeros((n_points, 2))
    for i in range(n_points - 1):
        (az12, az21, dist) = g.inv(lonlats[i, 0], lonlats[i, 1], lonlats[i + 1, 0], lonlats[i + 1, 1])
        velocities[i, :] = [speed * np.sin(az12), speed * np.cos(az12)]
    velocities[-1, :] = [speed * np.sin(az21), speed * np.cos(az21)]

    # Output array
    return np.concatenate((lonlats[:, [1, 0]], times[:, None], velocities), axis=1)


def load_MarineTraffic_csv(filepath):
    # filepath = './Data/exportvesseltrack477620900.csv'
    full = pd.read_csv(filepath, skipinitialspace=True)

    # Convert timestamp from string to Matlab datenum
    timestamp2matlabdn = lambda x: datetime2matlabdn(
        datetime.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S'))
    full['TIMESTAMP'] = full['TIMESTAMP'].apply(timestamp2matlabdn)

    full = full.sort_values(by='TIMESTAMP', ascending=False)

    # Now embarrassingly throw all the data away (for now)
    return full[['LAT', 'LON', 'TIMESTAMP']].values


def plot_path(ptv_matrix):
    # fig = plt.figure(figsize=(10, 5))
    # ax = plt.axes([0, 0, 1, 1], projection=ccrs.PlateCarree(central_longitude=180))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))

    # make the map global rather than have it zoom in to
    # the extents of any plotted data
    ax.set_global()

    # ax.stock_img()
    ax.coastlines()

    ax.plot(-0.08, 51.53, 'o', transform=ccrs.PlateCarree())
    ax.plot([-0.08, 132], [51.53, 43.17], transform=ccrs.PlateCarree())
    ax.plot([-0.08, 132], [51.53, 43.17], transform=ccrs.Geodetic())

    plt.show()


def folium_path(ptv_matrix):
    m = folium.Map([30, 0], zoom_start=3)

    folium.PolyLine(ptv_matrix[:, :2].tolist()).add_to(m)
    m.save('map.html')

    return m

# # point A: London
# startlat = 51.5074
# startlong = 0.1278
# starttime = datetime.datetime.now()
#
# # point B: NewYork
# endlat = 40.730610
# endlong = -73.935242
# endtime = starttime + datetime.timedelta(days=9)


