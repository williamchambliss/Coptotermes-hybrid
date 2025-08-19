import glob
import os
import math

import pandas as pd

import h5py

import numpy as np
import scipy
from scipy.interpolate import interp1d

#------------------------------------------------------------------------------#
def fill_missing(Y, kind="linear"):
    initial_shape = Y.shape
    Y = Y.reshape((initial_shape[0], -1))
    for i in range(Y.shape[-1]):
        y = Y[:, i]
        x = np.flatnonzero(~np.isnan(y))
        if len(x) > 3:
            f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)
            xq = np.flatnonzero(np.isnan(y))
            y[xq] = f(xq)
            mask = np.isnan(y)
            y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])
            Y[:, i] = y
    Y = Y.reshape(initial_shape)
    return Y
#------------------------------------------------------------------------------#

def data_filter(in_dir, treatment):
    df = pd.DataFrame()
    df_body = pd.DataFrame()
    files = glob.glob(in_dir + "/*.h5")

    for f_name in files:
        with h5py.File(f_name, "r") as f:
            dset_names = list(f.keys())
            locations = f["tracks"][:].T
            node_names = [n.decode() for n in f["node_names"][:]]

        # Check if more than 1 termite track exists
        if locations.shape[3] > 1:
            print("Warning: more than one termite track found, only using the first.")
            locations = locations[:, :, :, 0:1]  # Keep only the first termite

        locations = fill_missing(locations)

        # Apply median filter
        for i_coord in range(locations.shape[2]):
            for i_nodes in range(locations.shape[1]):
                locations[:, i_nodes, i_coord, 0] = scipy.signal.medfilt(
                    locations[:, i_nodes, i_coord, 0], 5
                )

        video = os.path.splitext(os.path.basename(f_name))[0]

        if 'abdomentip' not in node_names:
            raise ValueError("Error: 'abdomentip' is missing from node_names.")

        # --- Body length for single termite ---
        head_x = locations[:, node_names.index('headtip'), 0, 0]
        head_y = locations[:, node_names.index('headtip'), 1, 0]
        tip_x = locations[:, node_names.index('abdomentip'), 0, 0]
        tip_y = locations[:, node_names.index('abdomentip'), 1, 0]
        center_x = locations[:, node_names.index('abdomenfront'), 0, 0]
        center_y = locations[:, node_names.index('abdomenfront'), 1, 0]

        body_length = np.median(
            np.sqrt((head_x - center_x) ** 2 + (head_y - center_y) ** 2) +
            np.sqrt((tip_x - center_x) ** 2 + (tip_y - center_y) ** 2)
        )

        df_temp = {
            "video": video,
            "body_length": body_length
        }
        df_body = pd.concat([df_body, pd.DataFrame([df_temp])])

        print("locations.shape:", locations.shape)

        # --- Extract key points ---
        locations_abdomen = locations[:, node_names.index('abdomentip'), :, 0]
        locations_headtip = locations[:, node_names.index('headtip'), :, 0]
        locations_center = locations[:, node_names.index('abdomenfront'), :, 0]

        x = locations_center[:, 0]
        y = locations_center[:, 1]
        x_abdomen = locations_abdomen[:, 0]
        y_abdomen = locations_abdomen[:, 1]
        x_headtip = locations_headtip[:, 0]
        y_headtip = locations_headtip[:, 1]

        df_temp = pd.DataFrame({
            "video": video,
            "x": x,
            "y": y,
            "x_abdomen": x_abdomen,
            "y_abdomen": y_abdomen,
            "x_headtip": x_headtip,
            "y_headtip": y_headtip
        })

        df = pd.concat([df, df_temp])

    df_body.to_csv('D:/Coptotermes_hybrid/solo/cropped/data_fmt/' + treatment + '_bodysize.csv', index=False)
    return df
#------------------------------------------------------------------------------#

def main_data_filter():
    data_place = ["D:/Coptotermes_hybrid/solo/cropped/h5files"]
    for data_place_i in data_place:
        treatment = os.path.basename(data_place_i)
        df = data_filter(in_dir=data_place_i, treatment=treatment)
        filename = treatment + "_df.feather"
        df.reset_index().to_feather("D:/Coptotermes_hybrid/solo/cropped/data_fmt/" + filename)
#------------------------------------------------------------------------------#

if __name__ == "__main__":
    main_data_filter()

