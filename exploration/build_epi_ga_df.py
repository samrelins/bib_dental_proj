#%%

import pandas as pd
import numpy as np
import os


def build_epi_ga_data(data_dir):

    # read in GA data
    ga_data_path = os.path.join(data_dir, "ga_data.csv")
    ga_data = pd.read_csv(ga_data_path)

    # add nan values in place of blank string
    ga_data.replace([" "], np.nan, inplace=True)

    # weight variable is numeric but stored as object - convert to float
    ga_data["dgaWeight"] = ga_data.dgaWeight.astype("float")

    # correct dgaGA variables - have been incorrectly converted to odd categories
    # (no clear reason why this happened
    correct_tot_seq_vals = {
        "Extracted": 1,
        "Filled": 2,
        "Sealed": 3,
    }
    ga_data["dgaGATotal"] = ga_data["dgaGATotal"].map(correct_tot_seq_vals)
    ga_data["dgaGATotal"] = ga_data["dgaGATotal"].astype("int")
    ga_data["dgaGASequence"] = ga_data["dgaGASequence"].map(correct_tot_seq_vals)
    ga_data["dgaGASequence"] = ga_data["dgaGASequence"].astype("int")

    correct_type_vals = {
        "Extracted" : "exodontia",
        "Filled" : "comprehensive_care"
    }

    ga_data["dgaGAType"] = ga_data.dgaGAType.map(correct_type_vals)

    # read in merged data
    merge_data_path = os.path.join(data_dir, "epi_ga_merge_data.csv")
    merge_data = pd.read_csv(merge_data_path)
    merge_data.replace([" "], np.nan, inplace=True)

    # for some reason first two entries are blank
    merge_data = merge_data.iloc[2:]

    # convert ChildID to correct int type
    merge_data["ChildID"] = merge_data.ChildID.astype("int")

    # convert dgaGAType to category values
    type_keys_to_vals = {
        1 : "exodontia",
        2 : "comprehensive_care"
    }
    merge_data["dgaGAType"] = merge_data.dgaGAType.astype("float")
    merge_data["dgaGAType"] = merge_data.dgaGAType.map(type_keys_to_vals)

    # merge AgeMths col from GA data into merged data
    merge_data.drop("AgeMths", axis=1, inplace=True)
    key_cols = ["ChildID", "dgaGASequence"]
    merge_data = merge_data.merge(ga_data[key_cols + ["AgeMths"]],
                                  how="left",
                                  on=key_cols)

    # remove duplicate entries - might have arisen from fudged attempt to account
    # for kids that have been given multiple anaesthetics
    duplicate_ga_entries = ga_data.dgaGASequence == 2
    ga_data = ga_data[~duplicate_ga_entries]

    duplicate_merged_entries = merge_data.dgaGASequence == 2
    merge_data = merge_data[~duplicate_merged_entries]

    # swap out fudged Weigth variable for orig in GA data
    merge_data.drop("dgaWeight", axis=1, inplace=True)
    ga_key_cols = ["ChildID", "AgeMths"]
    merge_data = merge_data.merge(ga_data[ga_key_cols + ["dgaWeight"]],
                                  how="left",
                                  on=ga_key_cols)

    # swap mcage and mcbetterstart entries from GA data
    mc_cols = ["res6mcage", "res6mcbetterstart"]
    merge_data = merge_data.merge(ga_data[ga_key_cols + mc_cols],
                                  how="left",
                                  on=ga_key_cols,
                                  suffixes=("", "_ga"))
    mcage_is_na = merge_data.res6mcage.isna()
    merge_data.loc[mcage_is_na, "res6mcage"] = merge_data[mcage_is_na].res6mcage_ga

    bs_is_na = merge_data.res6mcbetterstart.isna()
    merge_data.loc[bs_is_na, "res6mcbetterstart"] = (
        merge_data[bs_is_na].res6mcbetterstart_ga
    )
    merge_data.drop("res6mcbetterstart_ga", axis=1, inplace=True)

    return merge_data

