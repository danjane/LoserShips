# Have a look at what we've got
#
# Work in Lon-Lat (x,y)

import boat
import satellite

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

import pickle


def p_value(mu1, n1, mu2, n2):
    # If X, Y have exponential distributions then muY/muX has an F distribution
    # So 1-sided p-value calculated from ratio of means
    #
    # https://stats.stackexchange.com/questions/76689/how-to-compare-the-mean-of-two-samples-whose-data-fits-exponential-distributions
    return scipy.stats.f.cdf(mu2 / mu1, 2 * n1, 2 * n2)


def p_value_of_smoke_distribution(so2_no_smoke, so2_smoke):
    return p_value(np.mean(so2_smoke), so2_smoke.shape[0], np.mean(so2_no_smoke), so2_no_smoke.shape[0])


def smoke(s_near, b_near, ds):
    # Identify pairs just after boat has passed
    smoke_idx = np.ones((b_near.shape[0],), dtype=bool)
    smoke_idx[ds > 50. * 1000.] = False
    smoke_idx[s_near[:, 2] - b_near[:, 2] < 0] = False  # boat is after satellite!
    smoke_idx[s_near[:, 2] - b_near[:, 2] > 1] = False  # more than a day later

    return smoke_idx


def plot(so2_no_smoke, so2_smoke, plot_name='test2.pdf'):
    # Plot the results
    plt.style.use('seaborn-deep')
    fig, ax = plt.subplots()

    n, bins, patches = plt.hist([so2_no_smoke, so2_smoke], 50, density=True, label=['no_smoke', 'smoke'])
    plt.legend(loc='upper right')

    ax.set_xlabel('SO2 detected in PBL')
    ax.set_ylabel('Probability density')
    pval = p_value_of_smoke_distribution(so2_no_smoke, so2_smoke)
    ax.set_title('Histogram of SO2 detected in PBL, p-value {0:.4f}'.format(pval))

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()

    plt.ion()
    plt.show()
    plt.pause(0.001)
    plt.savefig(plot_name)

    return pval


def reduce_amount_of_boat_and_satellite_data(ptv, s_pd, idx):
    nb_pairs = np.sum(idx)
    print("Number of boat-satellite position pairs: {0:.2g}".format(nb_pairs))

    bidx = np.any(idx, axis=0)
    sidx = np.any(idx, axis=1)

    ptv = ptv[bidx, :]
    s_pd = s_pd[sidx]
    idx_2d = np.ix_(sidx, bidx)
    return ptv, s_pd, idx_2d


if __name__ == "__main__":

    print("Loading data...")

    # Boat info: position(lonlat), time, velocity
    boat_filepath = './Data/exportvesseltrack477620900.csv'
    ptv = boat.load_MarineTraffic_csv(boat_filepath)
    #boat_filepath = './Data/vlcc_pacific_path.csv'
    #ptv = boat.load_UMAS_csv(boat_filepath)


    ptv = boat.interpolate_positions_times_velocities(ptv)

    # Satellite info:
    # # 'ColumnAmountSO2_PBL', 'Date', 'Latitude', 'Longitude',
    # # 'QualityFlags_PBL', 'RadiativeCloudFraction', 'SolarZenithAngle', 'Time'
    aura_filepath = '../LoserShips/backup2018.csv'
    s_pd = pd.read_csv(aura_filepath)
    print("Number of boat-satellite position pairs: {0:.2g}".format(ptv.shape[0] * s_pd.shape[0]))

    # Way too slow
    # ds = satellite.boat_satellite_distance(ptv[:, 0:2], s_pd[['Longitude', 'Latitude']].values)

    # # # # !!!! >>>>>>>>>>>>> Need to massively reduce ptv.shape[0] * s_pd.shape[0] <<<<<<<<<<<<< !!!! # # # #

    print("Throwing out error so2 readings...")
    chk = s_pd['ColumnAmountSO2_PBL'] >= 0
    s_pd = s_pd[chk]
    print("Number of boat-satellite position pairs: {0:.2g}".format(ptv.shape[0] * s_pd.shape[0]))

    # Avoid discontinuity in Longitude across international date boundary
    s_pd = satellite.pacific(s_pd)
    ptv = boat.pacific(ptv)

    max_dist = 50  # km
    print("Restricting to close ({}km) Lat pairs...".format(max_dist))

    # This is a '1d' reduction, keeping just the satellite data near the boat path
    s_pd = satellite.restrict_area_basic(s_pd, np.min(ptv[:, 0]), np.max(ptv[:, 0]), np.min(ptv[:, 1]),
                                             np.max(ptv[:, 1]))

    # First full blown 2d construction, so a massive array. Fasten your seat belts...
    dLats = np.abs(s_pd["Latitude"].values[:, None] - ptv[:, 1:2].T) * np.sqrt(satellite.LAT_ADJ2)
    ptv, s_pd, idx_2d = reduce_amount_of_boat_and_satellite_data(ptv, s_pd, dLats < max_dist * 1000.)
    dLats = dLats[idx_2d]

    # Take the pairs of points about max_dist (200km?) from each other
    print("Restricting to roughly nearby ({}km on flat Earth) pairs...".format(max_dist))
    dLons = np.abs(s_pd["Longitude"].values[:, None] - ptv[:, 0:1].T) * np.sqrt(satellite.EQUATOR_LON_ADJ2)
    dLons *= np.cos(np.deg2rad(s_pd["Latitude"].values[:, None]))
    ds = np.sqrt(dLons ** 2 + dLats ** 2)

    idx = ds < max_dist * 1000.
    ptv, s_pd, idx_2d = reduce_amount_of_boat_and_satellite_data(ptv, s_pd, idx)
    ds = ds[idx_2d]

    # Big step now, will list all the pairs...
    # ...this will (almost certainly) create duplicate rows of satellite and/or boat data, so could explode.
    idx = idx[idx_2d]
    (s_idx, b_idx) = idx.nonzero()
    assert (s_idx.shape[0] < 1e6), "Still too many pairs to handle!!"

    s_near = satellite.convert_satellite_pandas(s_pd.iloc[s_idx])
    ptv = ptv[b_idx, :]

    # Calculate geodesic distances for these pairs
    pickle.dump((ptv, s_near), open('ina.pickle', "wb"))
    lon_lats = np.concatenate((ptv[:, :2], s_near[:, :2]), axis=1)
    ds = np.array([satellite.geod_distance(*row) for row in lon_lats])
    dist_idx = np.argsort(ds)
    print("Shortest distance: {0:.2f}".format(ds[dist_idx[0]]))

    # Identify satellite readings taken just after the boat has passed
    smoke_idx = smoke(s_near, ptv, ds)
    num_smoke = np.sum(smoke_idx)
    num_no_smoke = ds.shape[0] - num_smoke

    print("Number of     smoke pairs: {}".format(num_smoke))
    print("Number of non-smoke pairs: {}".format(num_no_smoke))

    # General distribution of so2 values
    so2_smoke = s_near[smoke_idx, -1]
    so2_no_smoke = s_near[np.logical_not(smoke_idx), -1]

    if num_smoke * num_no_smoke:
        pval = plot(so2_no_smoke, so2_smoke)
        print("Distribution behind ship has different mean with p-value of {0:.4f}".format(pval))


