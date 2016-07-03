import glob
import numpy as np
import pandas as pd
import scipy.interpolate as interpolate

# Draw results
DEBUG = True

# route table
# split to different tracks in order to not overfit while testing
ROUTE_TABLE = [[0, 0, 1, 1], [2, 2, 3, 3, 4, 4], [5, 6], [7, 7, 8, 8, 9]]
# Tested route
ROUTE_TEST = [9]

# Write to file. None for no file written
output_prefix = None


# Find time limits per POSI id
def posi_time_limits(pos_array):
    # create start and end index time array
    pos_array['ind_begin_time'] = np.zeros((pos_array.shape[0]))
    pos_array['ind_end_time'] = np.zeros((pos_array.shape[0]))
    pos_array['ind_end_time'].iloc[-1] = 99999

    for i in range(1, pos_array.shape[0]):
        cur_split = (float(pos_array['Time'].iloc[i]) + float(pos_array['Time'].iloc[i-1])) / 2
        pos_array['ind_begin_time'].iat[i] = cur_split
        pos_array['ind_end_time'].iat[i-1] = cur_split
    return pos_array


# get train file names
def main(test_route):
    """
    Read files
    """
    route_folders = ['CAR', 'UAH', 'UJITI', 'UJIUB']

    files_names = []
    n_files = 0
    for building in route_folders:
        cur_file_names = glob.glob('routes/' + building + '/log*')
        files_names.append(sorted(cur_file_names))
        n_files += len(cur_file_names)

    print(files_names)
    print('There are %d files' % n_files)

    # import files into dataframes
    track = []
    for building in files_names:
        track.append([])
        for track_name in building:
            print(track_name)
            track[-1].append(pd.read_csv(track_name, sep=';', engine='c', names=range(11)))
            # print(np.unique(track[-1][-1][0]))

    """
    Ground Truth association
    """
    # create true ground dataframes
    pos = []
    for i in range(len(track)):
        pos.append([])
        for j in range(len(track[i])):
            pos[i].append(track[i][j][[1, 2, 3, 4, 5, 6, 7]].iloc[track[i][j][0].values == 'POSI'])
            pos[i][j] = pos[i][j].dropna(axis=1)
            pos[i][j].columns = ['Time', 'index', 'Lat', 'Lon', 'FloorID', 'BuildingID']
            pos[i][j] = posi_time_limits(pos[i][j])

    # associate signals with POSI ids. Spliting at middle of the time between POSI measurements using posi_time_limits
    for i in range(len(track)):
        for j in range(len(track[i])):
            track[i][j]['POSI_floor'] = np.zeros((track[i][j].shape[0])) * np.nan
            track[i][j]['POSI_building'] = np.zeros((track[i][j].shape[0])) * np.nan
            for k in range(pos[i][j].shape[0]):
                ind_begin_time = pos[i][j]['ind_begin_time'].iloc[k]
                ind_end_time = pos[i][j]['ind_end_time'].iloc[k]

                ind_floor = pos[i][j]['FloorID'].iloc[k]
                ind_building = pos[i][j]['BuildingID'].iloc[k]

                index_limits = np.logical_and(track[i][j][1].values >= ind_begin_time,
                                              track[i][j][1].values < ind_end_time)

                track[i][j]['POSI_floor'].iloc[index_limits] = ind_floor
                track[i][j]['POSI_building'].iloc[index_limits] = ind_building

    # Pad ground truth in order to use interpolation later
    for i in range(len(track)):
        for j in range(len(track[i])):
            pos_pad_start = pos[i][j].iloc[0]
            pos_pad_start.at['Time'] = 0
            pos_pad_end = pos[i][j].iloc[-1]
            pos_pad_end.at['Time'] = track[i][j][1].values[-1]
            pos_array = np.vstack((pos_pad_start.values.reshape((1, pos[i][j].shape[1])),
                                   pos[i][j].values,
                                   pos_pad_end.values.reshape((1, pos[i][j].shape[1]))))
            pos[i][j] = pd.DataFrame(pos_array, columns=pos[i][j].columns.values)

    i = 1
    j = 2
    print(files_names[i][j], i, j)
    print(pos[i][j].head())

    # Interpolate sensors' time
    for i in range(len(track)):
        for j in range(len(track[i])):
            interp_ground_truth_lat = interpolate.interp1d(pos[i][j]['Time'].astype(float), pos[i][j]['Lat'])
            track[i][j]['interp_lat'] = interp_ground_truth_lat(track[i][j][1])
            interp_ground_truth_lon = interpolate.interp1d(pos[i][j]['Time'].astype(float), pos[i][j]['Lon'])
            track[i][j]['interp_lon'] = interp_ground_truth_lon(track[i][j][1])
    print(track[0][0].head())

    """
    WiFi
    """
    wifi = []
    for i in range(len(track)):
        wifi.append([])
        for j in range(len(track[i])):
            wifi[-1].append(track[i][j][[1, 4, 5, 'POSI_floor', 'POSI_building', 'interp_lat',
                                         'interp_lon']].iloc[track[i][j][0].values == 'WIFI'])
            wifi[-1][-1].columns = ['AppTime', 'MAC', 'rssi', 'POSI_floor', 'POSI_building', 'interp_lat', 'interp_lon']
    print(wifi[0][0].head())

    # Only calculate for the train data
    # create list of mac addresses for all the samples
    total_macs = []
    print('total macs')
    for i in range(len(wifi)):
        for j in range(len(wifi[i])):
            if ROUTE_TABLE[i][j] != test_route:
                track_mac_array = wifi[i][j]['MAC'].values
                for mac_add in track_mac_array:
                    if not mac_add in total_macs:
                        total_macs.append(mac_add)
    print(len(total_macs))

    # create list of mac addresses for each building
    building_macs = {10: [], 20: [], 30: [], 40: []}
    for i in range(len(wifi)):
        for j in range(len(wifi[i])):
            if ROUTE_TABLE[i][j] != test_route:
                track_mac_array = wifi[i][j]
                for k in range(track_mac_array.shape[0]):
                    mac_line = track_mac_array.iloc[k]
                    mac = mac_line['MAC']
                    build = int(mac_line['POSI_building'])
                    if not mac in building_macs[build]:
                        building_macs[build].append(mac)
    print('building macs')
    for build in sorted(building_macs):
        print(build, len(building_macs[build]), ',')

    # create list of mac addresses for each building for each floor
    floor_macs = {10: [[], [], [], [], [], []], 20: [[], [], [], [], [], []],
                  30: [[], [], [], [], [], []], 40: [[], [], [], [], [], []]}
    for i in range(len(wifi)):
        for j in range(len(wifi[i])):
            if ROUTE_TABLE[i][j] != test_route:
                track_mac_array = wifi[i][j]
                for k in range(track_mac_array.shape[0]):
                    mac_line = track_mac_array.iloc[k]
                    mac = mac_line['MAC']
                    build = int(mac_line['POSI_building'])
                    floor = int(mac_line['POSI_floor'])
                    if not mac in floor_macs[build][floor]:
                        floor_macs[build][floor].append(mac)

    print('floor macs')
    for build in sorted(floor_macs):
        for j in range(len(floor_macs[build])):
            print(build, j, len(floor_macs[build][j]), ',')

main(ROUTE_TEST)
