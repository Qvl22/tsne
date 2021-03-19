import pandas as pd
import numpy as np
import os
import math
import gpxpy
import scipy.spatial.distance
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from geographiclib.geodesic import Geodesic


# Func to convert gpx file into pandas Data Frame
def gpx_to_df(folder):
    list_of_files = os.listdir(folder)
    dataframe_list = []
    for file in list_of_files:
        gpx_file = gpxpy.parse(open('{}/{}'.format(folder, file)))
        if len(gpx_file.tracks) != 0:
            segment = gpx_file.tracks[0].segments[0]
            df = pd.DataFrame([
                {'time': p.time,
                 'lat': p.latitude,
                 'long': p.longitude,
                 'ele': p.elevation} for p in segment.points])
            dataframe_list.append(df)
    return dataframe_list


def geo_to_decart(lon, lat, h):
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b ** 2 / a ** 2)
    n = float(a / math.sqrt(1 - e2 * (math.sin(math.radians(abs(lat))) ** 2)))
    x = (n + h) * math.cos(math.radians(lat)) * math.cos(math.radians(lon))
    y = (n + h) * math.cos(math.radians(lat)) * math.sin(math.radians(lon))
    return x, y


# Func to calculate the distance between points and write it to new column 'distance'
def calculate_distance(dl):
    df_list = []
    for df in dl:
        arr = df.to_numpy()
        num_rows, num_cols = arr.shape
        temp_dist = np.zeros((num_rows, 1))
        temp_brng = np.zeros((num_rows, 1))
        if num_cols == 4:
            for row in range(num_rows):
                if row != 0 and row + 1 < num_rows:
                    brng = Geodesic.WGS84.Inverse(arr[row, 1], arr[row, 2], arr[row+1, 1], arr[row+1, 2])['azi1']
                    x, y = geo_to_decart(arr[row, 1], arr[row, 2], arr[row, 3])
                    a, b = geo_to_decart(arr[row + 1, 1], arr[row + 1, 2], arr[row, 3])
                    distance = ((x - a) ** 2 + (y - b) ** 2) ** (1 / 2)
                    temp_dist[row] = distance
                    temp_brng[row] = brng
                else:
                    temp_dist[0] = 0
                    temp_brng[0] = 0
            arr = np.column_stack((arr, temp_dist))
            arr = np.column_stack((arr, temp_brng))
            df_list.append(pd.DataFrame(arr, columns=['time', 'latitude', 'longitude',
                                                      'altitude', 'distance', 'bearing']))
            df_list[len(df_list) - 1].set_index('time', inplace=True)
    return df_list


# Func to turn list of data series into data frame
def to_table(dl):
    df = pd.DataFrame()
    for ds in dl:
        ds = ds.reset_index(drop=True)
        ds = ds.rename_axis('{}'.format(df.shape[1] - 1))
        df = pd.concat([df, ds], axis=1)
        df = df.rename(columns=lambda x: '{}'.format(df.columns.get_loc(x)))
    if df.shape[0] < df.shape[1]:
        for i in range(df.shape[1] - df.shape[0]):
            df = df.append(pd.Series(dtype=float), ignore_index=True)
    df = df.fillna(0)
    return df


# Func to create a similarity table for list of data series
def similarity_table(df):
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    m_euclid = np.zeros((df.shape[1], df.shape[1]))
    for i in range(df.shape[1]):
        u = df[df.columns[i]]
        for j in range(df.shape[1]):
            v = df[df.columns[j]]
            m_euclid[i, j] = round((1 / (1 + scipy.spatial.distance.euclidean(u, v))), 5)
    print(m_euclid)
    df_euclid = pd.DataFrame(m_euclid, columns=df.columns, index=df.columns)
    return df_euclid


# Func to create a Series list from a Data Frame list by using some column "name" and resampling it by time period of t
def create_series_list(dl, name, t, f):
    distance_series = []
    for i in dl:
        distance_series.append(i[name])
        if f == 'sum':
            distance_series[len(distance_series) - 1] = distance_series[len(distance_series) - 1].resample(t).sum()
        elif f == 'mean':
            distance_series[len(distance_series) - 1] =\
                distance_series[len(distance_series) - 1].astype(float).resample(t).mean()
        else:
            print("Invalid frequency")
    return distance_series


def t_sne(df, name, clr):
    x = np.vstack([df['{}'.format(i)]
                   for i in range(df.shape[1])])
    tsne_features = TSNE(random_state=10).fit_transform(x)
    sns.scatterplot(x=tsne_features[:, 0], y=tsne_features[:, 1], data=df.T, color=clr, label=name)


def classic_algo(name):
    dl = gpx_to_df('{}_activities'.format(name))
    dl = calculate_distance(dl)
    speed_df = create_series_list(dl, 'distance', 'min', 'sum')
    speed_df = to_table(speed_df)
    print('Similarity matrix of speed for {}'.format(name.capitalize()))
    similarity_table(speed_df)
    bearing_df = create_series_list(dl, 'bearing', 'min', 'sum')
    bearing_df = to_table(bearing_df)
    print('\n\nSimilarity matrix of bearing for {}'.format(name.capitalize()))
    similarity_table(bearing_df)
    if name == 'anthony':
        c = 'red'
    elif name == 'zach':
        c = 'blue'
    t_sne(speed_df, '{}_activities'.format(name.capitalize()), c)


if __name__ == '__main__':

    classic_algo('zach')
    classic_algo('anthony')

    plt.show()
