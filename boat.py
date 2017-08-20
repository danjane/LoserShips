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

# https://stackoverflow.com/questions/8776414/python-datetime-to-matlab-datenum
def datetime2matlabdn(dt):
    mdn = dt + datetime.timedelta(days=366)
    frac_seconds = (dt-datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0)).seconds / (24.0 * 60.0 * 60.0)
    frac_microseconds = dt.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
    return mdn.toordinal() + frac_seconds + frac_microseconds


# point A: London
startlat = 51.5074
startlong = 0.1278
starttime = datetime.datetime.now()

# point B: NewYork
endlat = 40.730610
endlong = -73.935242
endtime = starttime + datetime.timedelta(days=9)


########### Intermediate locations
g = pyproj.Geod(ellps='WGS84')
(az12, az21, dist) = g.inv(startlong, startlat, endlong, endlat)

dist /= 1000. #switch to km
# calculate line string along path with segments <= 1 km
lonlats = g.npts(startlong, startlat, endlong, endlat,
                 1 + int(dist))

# npts doesn't include start/end points, so prepend/append them
lonlats.insert(0, (startlong, startlat))
lonlats.append((endlong, endlat))
lonlats = np.asarray(lonlats) # warning! still y-x (=lon,lat)


########### Intermediate times
d_time = endtime - starttime
n_points = lonlats.shape[0]
i_time = d_time / (n_points - 1)
times = np.zeros((n_points,1))
for i in range(n_points):
    times[i] = datetime2matlabdn(starttime + i_time*i)


########### Intermediate velocities
speed = dist / (times[-1]-times[0]) / 24. #currently assuming constant speed
velocities = np.zeros((n_points, 2))
for i in range(n_points-1):
    (az12, az21, dist) = g.inv(lonlats[i, 0], lonlats[i, 1], lonlats[i+1, 0], lonlats[i+1, 1])
    velocities[i, :] = [speed * np.sin(az12), speed * np.cos(az12)]
velocities[-1, :] = [speed * np.sin(az21), speed * np.cos(az21)]

########### Output array
location_time_velocity = np.concatenate((lonlats[:, [1, 0]], times, velocities), axis=1)
