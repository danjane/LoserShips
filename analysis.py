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

# Way too slow
# ds = satellite.boat_satellite_distance(ptv[:, 0:2], s_pd[['Longitude', 'Latitude']].values)

# Identify useful satellite readings
# Nearby Lat readings
s_pd_box = satellite.restrict_area_basic(s_pd, np.min(ptv[:, 0]), np.max(ptv[:, 0]), np.min(ptv[:, 1]), np.max(ptv[:, 1]))
dLats = np.abs(s_pd_box["Latitude"].values[:, None] - ptv[:, 1:2].T) * np.sqrt(satellite.LAT_ADJ2)
idx = dLats < 200. * 1000.
bidx = np.any(idx, axis=0)
sidx = np.any(idx, axis=1)
print("Keeping {0:.2f}% of satellite readings (Latitude nearby)".format(100*sum(sidx)/num_satellite_readings))
s_pd_Lat = s_pd_box[sidx]
dLats = dLats[np.ix_(sidx, bidx)]
idx = idx[np.ix_(sidx, bidx)]
ptv = ptv[bidx, :]

# Now use Longitude as well for a rough Euclidean metric
dLons = np.abs(s_pd_Lat["Longitude"].values[:, None] - ptv[:, 0:1].T) * np.sqrt(satellite.EQUATOR_LON_ADJ2)
dLons *= np.cos(np.deg2rad(s_pd_Lat["Latitude"].values[:, None]))
ds = np.sqrt(dLons**2 + dLats**2)

# Take the pairs of points about 200km from each other
idx = ds < 200. * 1000.
(s_idx, b_idx) = idx.nonzero()
s_near = satellite.convert_satellite_pandas(s_pd_Lat.iloc[s_idx])

# Calculate geodesic distances for these pairs
lon_lats = np.concatenate((ptv[b_idx, 0:2], s_pd_near[['Longitude', 'Latitude']].values), axis=1)
ds = np.array([satellite.geod_distance(*row) for row in lon_lats])

# Identify pairs just after boat has passed
smoke_idx = np.ones((s_idx.shape[0],), dtype=bool)
smoke_idx[s_near[:, 2] < ptv[b_idx, 2]] = False
smoke_idx[ds > 50000] = False
smoke_idx[s_near[:, 2] - ptv[b_idx, 2] > 1] = False

# General distribution of so2 values
so2_no_smoke = s_pd_Lat.iloc[np.unique(s_idx[np.logical_not(smoke_idx)])]['ColumnAmountSO2_PBL'].values
so2_smoke = s_pd_Lat.iloc[np.unique(s_idx[smoke_idx])]['ColumnAmountSO2_PBL'].values

# Plot the results
plt.style.use('seaborn-deep')
fig, ax = plt.subplots()

n, bins, patches = plt.hist([so2_no_smoke, so2_smoke], 50, density=True, label=['no_smoke', 'smoke'])
plt.legend(loc='upper right')

# ax.plot(s_closest['ColumnAmountSO2_PBL'] * np.ones((2, 1)), np.array([0, 0.9*top]), 'r--')
# plt.text(s_closest['ColumnAmountSO2_PBL'], 0.95*top, 'after ship', color='r', horizontalalignment='center')
#
# ax.plot(np.mean(so2) * np.ones((2, 1)), np.array([0, 0.9*top]), 'k--')
# plt.text(np.mean(so2), 0.95*top, 'mean', horizontalalignment='center')

ax.set_xlabel('SO2 detected in PBL')
ax.set_ylabel('Probability density')
ax.set_title('Histogram of SO2 detected in PBL')

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()

plt.ion()
plt.show()
plt.pause(0.001)
plt.savefig('test.pdf')

