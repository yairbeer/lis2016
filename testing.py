import numpy as np


def goal(labels, predictions):
    latlon_err = np.sqrt((labels['Lat'].values - predictions['Lat'].values) ** 2 +
                         (labels['Lon'].values - predictions['Lon'].values) ** 2)
    # change to meters
    latlon_err *= 10000
    return 0