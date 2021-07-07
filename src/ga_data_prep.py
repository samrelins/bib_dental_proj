
import pandas as pd
import numpy as np


def list_tooth_colnames(df):
    df = df.copy()
    tooth_cols = [col for col in df.columns
                  if col[-3:-1] in ["UR", "UL", "LR", "LL"]]
    if len(tooth_cols) < 34:
        print("WARNING: Looks like some columns may have been renamed")
        print("This helper only works with the original GA column names")

    return tooth_cols


def clean_ga_data(df):
    df = df.copy()

    # fill all tooth nans with zeros (no treatment)
    tooth_cols = list_tooth_colnames(df)

    for col in tooth_cols:
        df[col] = df[col].fillna(0)

    # the first and second entries for all observations with 3
    # GAs are duplicates - remove
    has_3_gas = df.GATotal == 3
    seq_is_2 = df.GASequence == 2
    remove_cols = has_3_gas & seq_is_2
    df = df[~remove_cols]

    # correct observations with 3 GAs in sequence
    seq_is_3 = df.GASequence == 3
    df.loc[has_3_gas, "GATotal"] = 2
    df.loc[seq_is_3, "GASequence"] = 2

    return df


def rename_ga_columns(df):
    df = df.copy()

    new_ga_colnames = {'AgeMths': "age_at_ga", 'GAType': "ga_type",
                       'GATotal': "total_gas", 'GASequence': "ga_sequence",
                       'Weight': "weight_at_ga", 'dgaUR6': "ur6", 'dgaURE': "ure",
                       'dgaURD': "urd", 'dgaURC': "urc", 'dgaURB': "urb",
                       'dgaURA': "ura", 'dgaULA': "ula", 'dgaULB': "ulb",
                       'dgaULC': "ulc", 'dgaULD': "uld", 'dgaULE': "ule",
                       'dgaUL6': "ul6", 'dgaLL6': "ll6", 'dgaLLE': "lle",
                       'dgaLLD': "lld", 'dgaLLC': "llc", 'dgaLLB': "llb",
                       'dgaLLA': "lla", 'dgaLRA': "lra", 'dgaLRB': "lrb",
                       'dgaLRC': "lrc", 'dgaLRD': "lrd", 'dgaLRE': "lre",
                       'dgaLR6': "lr6", 'dgaUR4': "ur4", 'dgaUR1': "ur1",
                       'dgaUL1': "ul1", 'dgaUL4': "ul4", 'dgaLL5': "ll5",
                       'dgaLL4': "ll4", 'dgaLL1': "ll1", 'dgaLR1': "lr1",
                       'dgaLR4': "lr4", 'dgaLR5': "lr5"}

    return df.rename(new_ga_colnames, axis=1)


def convert_ga_numerical_categories(df):
    df = df.copy()

    tooth_treatment_map = {
        0: "nil_treatment",
        1: "extracted",
        2: "filled",
        3: "sealed",
        4: "crowned"
    }

    tooth_cols = list_tooth_colnames(df)

    for tooth_col in tooth_cols:
        df[tooth_col] = (df[tooth_col]
                         .map(tooth_treatment_map)
                         .astype("category"))

    ga_type_map = {
        1: "exodontia",
        2: "comprehensive_care"
    }
    df["GAType"] = (df.GAType
                    .map(ga_type_map)
                    .astype("category"))

    return df


def add_counts_of_treatments(df):
    df = df.copy()

    tooth_cols = list_tooth_colnames(df)

    # add count of baby teeth extracted
    baby_teeth = [tooth for tooth in tooth_cols
                  if tooth[-1] in ["A", "B", "C", "D", "E"]]
    n_primary_extractions = np.zeros(len(df))
    for tooth in baby_teeth:
        n_primary_extractions += df[tooth] == 1
    df["n_primary_extractions"] = n_primary_extractions

    # add count of perm teeth extracted
    perm_teeth = [tooth for tooth in tooth_cols
                  if tooth[-1] not in ["A", "B", "C", "D", "E"]]
    n_perm_extractions = np.zeros(len(df))
    for tooth in perm_teeth:
        n_perm_extractions += df[tooth] == 1
    df["n_perm_extractions"] = n_perm_extractions

    # add total counts of treatments
    tooth_treatment_map = {
        1: "extracted",
        2: "filled",
        3: "sealed",
        4: "crowned"
    }
    df["n_treated"] = np.zeros(len(df))
    for treatment_code, treatment_name in tooth_treatment_map.items():
        colname = "n_" + treatment_name
        df[colname] = np.zeros(len(df))
        for col in tooth_cols:
            df[colname] += df[col] == treatment_code
        if treatment_name != "extracted":
            df["n_treated"] += df[colname]

    if sum(df["n_extracted"]) == 0:
        raise ValueError("Numeric category codes have been converted"
                         " - generating treatment counts requires tooth category codes."
                         "\nReorder pipeline to convert codes after creating"
                         " treatment counts")

    return df


def generate_tooth_level_data(df):
    
    # define tooth data and different treatment categories
    tooth_cols = [col for col in df.columns
                  if col[0] in ["u", "l"]]
                
    treatments = ["extracted", "filled", "sealed", "crowned"]
    
    # count up the different treatments by tooth and store in dict
    treatments_by_tooth = []
    for tooth in tooth_cols:
        tooth_data = {"tooth": tooth.upper()}
        for treatment in treatments:
            tooth_data["n_" + treatment.lower()] = sum(df[tooth] == treatment)
        treatments_by_tooth.append(tooth_data)

    # convert tooth dict to dataframe
    tooth_df = pd.DataFrame(treatments_by_tooth)
    
    # add total treatment count and order dataframe by most - least treated
    treatments = ["n_extracted", "n_filled", "n_sealed", "n_crowned"]
    tooth_df["n_treated"] = tooth_df[treatments].sum(axis=1)
    tooth_df.sort_values("n_treated",
                         inplace=True,
                         ascending=False)
    
    # add non extraction treatment feature
    non_extractions = ["n_sealed", "n_filled", "n_crowned"]
    tooth_df["n_non_extractions"] = tooth_df[non_extractions].sum(axis=1)
    
    return tooth_df
