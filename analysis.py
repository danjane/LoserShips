# Have a look at what we've got
#
# Work in Lon-Lat (x,y)

import boat
import satellite

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Boat info: position (lonlat), time, velocity
boat_filepath = './Data/exportvesseltrack477620900.csv'
ptv = boat.load_MarineTraffic_csv(boat_filepath)
ptv = boat.pacific(ptv)
ptv = boat.interpolate_positions_times_velocities(ptv)

# Satellite info:
# # 'ColumnAmountSO2_PBL', 'Date', 'Latitude', 'Longitude',
# # 'QualityFlags_PBL', 'RadiativeCloudFraction', 'SolarZenithAngle', 'Time'
aura_filepath = '../LoserShips/backup2018.csv'
s_pd = pd.read_csv(aura_filepath)
num_satellite_readings = s_pd.shape[0]
chk = s_pd['ColumnAmountSO2_PBL'] >= 0
s_pd = s_pd[chk]
s_pd = satellite.pacific(s_pd)
print("Keeping {0:.2f}% of satellite readings (non-negative)".format(s_pd.shape[0]/num_satellite_readings))
num_satellite_readings = s_pd.shape[0]

# Identify useful satellite readings
s_pd = satellite.restrict_area_basic(s_pd, np.min(ptv[:, 0]), np.max(ptv[:, 0]), np.min(ptv[:, 1]), np.max(ptv[:, 1]))
ds = np.abs(s_pd["Latitude"].values[:, None] - ptv[:, 1:2].T) * np.sqrt(satellite.LAT_ADJ2)
idx = ds < 2000. * 1000.
sidx = np.any(idx, axis=1)
print("Keeping {0:.2f}% of satellite readings (Latitude nearby)".format(100*sum(sidx)/num_satellite_readings))
s_pd = s_pd[sidx]
ds = ds[sidx, :]
idx = idx[sidx, :]

ds = np.abs(s_pd["Latitude"].values[:, None] - ptv[:, 1:2].T)

# Way too slow
# ds = satellite.boat_satellite_distance(ptv[:, 0:2], s_pd[['Longitude', 'Latitude']].values)

ss = satellite.convert_satellite_pandas(s_pd)



ds = satellite.boat_satellite_distance(ptv, ss)
pos = satellite.find_min_idx(ds)
s_closest, so2, idx = satellite.investigate_interesting_boat_position(ptv[pos[0], :], s_pd, pos[1])

# Plot the results
fig, ax = plt.subplots()

n, bins, patches = plt.hist(so2, 50, density=True, facecolor='green')
top = max(n)

ax.plot(s_closest['ColumnAmountSO2_PBL'] * np.ones((2, 1)), np.array([0, 0.9*top]), 'r--')
plt.text(s_closest['ColumnAmountSO2_PBL'], 0.95*top, 'after ship', color='r', horizontalalignment='center')

ax.plot(np.mean(so2) * np.ones((2, 1)), np.array([0, 0.9*top]), 'k--')
plt.text(np.mean(so2), 0.95*top, 'mean', horizontalalignment='center')

ax.set_xlabel('SO2 detected in PBL')
ax.set_ylabel('Probability density')
ax.set_title('Histogram of SO2 detected in PBL\naround Lon {0:.2f}, Lat {1:.2f}'.format(
    s_closest['Longitude'], s_closest['Latitude']
))

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()

plt.ion()
plt.show()
plt.pause(0.001)

