
# Need username and password defined in credentials.py
import credentials

import datetime
import numpy as np
import pandas as pd
import numpy.matlib
import pydap.cas.urs
import urllib2
import bs4
import re
import os
import time


def nasa_data_tolist(d):
    d = d.reshape(nT * 21, 1)
    d = d.squeeze().tolist()
    return d


def nasa_data_from_dataset(dataname):
    d = dataset[dataname][:, 20:41].data
    return nasa_data_tolist(d)


def nasa_times_from_dataset():
    t = dataset.SecondsInDay[:].data
    # need to replicate the time for each track in the sweep
    t = np.transpose(np.matlib.repmat(t, 21, 1))
    return nasa_data_tolist(t)


filename = 'so2data_hot_2014.csv'
filename_backup = 'backup2014.csv'


# Pick up where we left off if possible
if os.path.isfile(filename_backup):
    df_year = pd.read_csv(filename_backup)
    df_year['Date'] = [datetime.datetime.strptime(dstr, "%Y-%m-%d").date() for dstr in df_year['Date']]
    firstday = (max(df_year['Date'])-datetime.date(2014, 1, 1)).days + 1
else:
    df_year = pd.DataFrame({'Date': []})
    firstday = 0

# connect to a URL
url_year = 'https://aura.gesdisc.eosdis.nasa.gov/opendap/Aura_OMI_Level2/OMSO2.003/2014/'
for day in range(firstday, 365):
    date = datetime.date(2014, 1, 1) + datetime.timedelta(days=day)
    url = url_year + str(day+1).zfill(3) + '/'

    print '{}\n{}'.format(date, url)
    resp = urllib2.urlopen(url)

    # Use BeautifulSoup to parse the html looking for the he5.html links
    soup = bs4.BeautifulSoup(resp, from_encoding=resp.info().getparam('charset'))
    links = []
    for link in soup.find_all('a', href=lambda href: href and re.compile('he5\.html').search(href)):
        links.append(link.getText())

    # Just need a unique set of links, without the html orphan
    links = set(links) - set([u'html'])

    # Bring in all the data for this day
    dfs = []
    for link in links:
        he5url = url+link

        backoff_delay = 1
        while backoff_delay < 200:
            try:
                print 'Setting up authentication for {}'.format(link)
                session = pydap.cas.urs.setup_session(credentials.username, credentials.password, check_url=he5url)

                print('Connecting to data source...')
                dataset = pydap.client.open_url(he5url, session=session)
                nT, nXtracks = dataset['ColumnAmountSO2_PBL'].shape
                assert nT * nXtracks > 0, 'No data!'

                # Download data and arrange in a DataFrame
                datakeys = ['ColumnAmountSO2_PBL', 'Latitude', 'Longitude',
                            'RadiativeCloudFraction', 'SolarZenithAngle', 'QualityFlags_PBL']

                print 'Downloading data:'
                #print '\tTime...'
                datadict = {'Time': nasa_times_from_dataset()}
                for datakey in datakeys:
                    #print '\t{}...'.format(datakey)
                    datadict[datakey] = nasa_data_from_dataset(datakey)

                df = pd.DataFrame(datadict)

                # Apply basic data checks and keep a subset of the rows
                # chk = ((qFlag & 1) < 1) & (rcf < 0.2) & (sza < 40)
                chk = (df.RadiativeCloudFraction < 0.3) & (df.SolarZenithAngle < 60) & (df.ColumnAmountSO2_PBL > 0)
                df = df[chk]

                dfs.append(df)
                print 'Done! {} good records of {}'.format(sum(chk), chk.size)


            except:
                time.sleep(backoff_delay)
                backoff_delay *= 2

            else:
                break

    df_oneday = pd.concat(dfs, ignore_index=True)
    df_oneday['Date'] = date

    df_year = pd.concat([df_year, df_oneday], ignore_index=True)
    df_year.to_csv(filename_backup, index=False)

# Now store a more manageable csv of the hot results
# Just the high readings
chk = df_year['ColumnAmountSO2_PBL'] > 2
df_year = df_year[chk]

# Round unnec precision
df_year['Time'] = df_year['Time'].astype(int) #seconds
df_year['ColumnAmountSO2_PBL'] = df_year['ColumnAmountSO2_PBL'].round(1) #noise is more than 0.5DU anyway
df_year['Latitude'] = df_year['Latitude'].round(4) #4dp more than enough
df_year['Longitude'] = df_year['Longitude'].round(4) #4dp more than enough

df_year[['Date', 'Time', 'Latitude', 'Longitude', 'ColumnAmountSO2_PBL']].to_csv(filename, index=False)
