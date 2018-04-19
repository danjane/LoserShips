# Have a look at what we've got

import boat
import satellite

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

boat_filepath = './Data/exportvesseltrack477620900.csv'
ptv = boat.load_MarineTraffic_csv(boat_filepath)

aura_filepath = './Data/OMI_processed.csv'
s_pd = pd.read_csv(aura_filepath)
chk = s_pd['ColumnAmountSO2_PBL'] >= 0
s_pd = s_pd[chk]

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

plt.savefig('test_so2distribution.pdf')

plt.ion()
plt.show()
plt.pause(0.001)

