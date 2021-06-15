#%%

import pandas as pd
import numpy as np
import os

def build_epi_ga_data(data_dir, include_ids=False):

    # read in GA data
    ga_data_path = os.path.join(data_dir, "ga_data.csv")
    ga_data = pd.read_csv(ga_data_path)

    epi_data_path = os.path.join(data_dir, "epi_data.csv")
    epi_data = pd.read_csv(epi_data_path)

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

    # fill na cols for tooth level treatments with "Nil treatment"
    tooth_cols = [col for col in merge_data.columns
                  if col[:4] in ["dgaU", "dgaL"]]
    for col in tooth_cols:
        merge_data[col] = merge_data[col].fillna("Nil treatment")

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

    # correct Totalnoextractions
    extractions_sum = (
            merge_data.totalnumberofprimaryextractions +
            merge_data.totalnumberofpermanentextractions
    )
    merge_data["Totalnoextractions"] = extractions_sum

    # split ga and epi into separate dataframes and rename features
    merge_cols = [col for col in merge_data.columns
                  if col not in ga_data.columns
                  and col not in epi_data.columns]

    is_epi = merge_data.Epi == 1
    is_ga = ~merge_data.AgeMths.isna()

    ga_data_final = merge_data[is_ga][list(ga_data.columns) + merge_cols]
    epi_data_final = merge_data[is_epi][list(epi_data.columns) + merge_cols]

    rename_cols = {
        'admincgender': 'gender',
        'agefm_fbqall': 'father_age_at_q',
        'agemm_mbqall': 'mother_age_at_q',
        'ben0mentst': 'on_benefits',
        'bib6a02': 'describe_health_q',
        'bib6b04': 'has_diagnosis_q',
        'bib6b05': 'hospital_admission_q',
        'deminfeth3gpcomb': 'ethnicity',
        'edcont_eal': 'english_additional_lang',
        'edcont_lac': 'looked_after_child',
        'edcont_sen': 'special_ed_needs',
        'edu0fthede': 'father_highest_ed',
        'edu0mumede': 'mother_highest_ed',
        'eth0eth9gp': 'mother_ethnicity',
        'fbqageeduc': 'age_father_complete_ed',
        'fbqcountrybirth': 'father_birthplace',
        'imd_2010_decile_nat': 'imd_2010_decile',
        'mbqlcasep5gp': 'socio_economic_pos',
        'org0agemuk': 'age_mother_moved_uk',
        'org0mmubct': 'mother_birthplace',
        'qad0langua': 'questionaire_language',
        'Totalnoextractions': 'n_extractions',
        'totalnumberofprimaryextractions': 'n_primary_extract',
        'totalnumberofpermanentextractions': 'n_perm_extract',
        'AgeMths': 'age_at_ga',
        'ddsdt': 'decayed_teeth',
        'ddsmt': 'missing_teeth',
        'ddsft': 'filled_teeth',
        'ddsdmft': 'dmft',
        'dgaGAType': 'type_of_ga',
        'dgaGATotal': 'total_ga',
        'dgaGASequence': 'ga_sequence',
        'dgaWeight': 'weight_at_ga',
        'dgaUR6': 'ur6', 'dgaURE': 'urE', 'dgaURD': 'urd', 'dgaURC': 'urc',
        'dgaURB': 'urb', 'dgaURA': 'ura', 'dgaULA': 'ula', 'dgaULB': 'ulb',
        'dgaULC': 'ulc', 'dgaULD': 'uld', 'dgaULE': 'ule', 'dgaUL6': 'ul6',
        'dgaLL6': 'll6', 'dgaLLE': 'lle', 'dgaLLD': 'lld', 'dgaLLC': 'llc',
        'dgaLLB': 'llb', 'dgaLLA': 'lla', 'dgaLRA': 'lra', 'dgaLRB': 'lrb',
        'dgaLRC': 'lrc', 'dgaLRD': 'lrd', 'dgaLRE': 'lre', 'dgaLR6': 'lr6',
        'dgaUR4': 'ur4', 'dgaUR1': 'ur1', 'dgaUL1': 'ul1', 'dgaUL4': 'ul4',
        'dgaLL5': 'll5', 'dgaLL4': 'll4', 'dgaLL1': 'll1', 'dgaLR1': 'lr1',
        'dgaLR4': 'lr4', 'dgaLR5': 'lr5'
    }

    epi_data_final.rename(rename_cols, axis=1, inplace=True)
    ga_data_final.rename(rename_cols, axis=1, inplace=True)

    # add features that total each treatment type (extracted already there)
    tooth_cols = [col for col in ga_data_final.columns
                  if len(col) == 3 and col != "Epi"]
    treatments = ["Filled", "Sealed", "Crowned"]
    ga_data_final["n_treated"] = np.zeros(len(ga_data_final))
    for treatment in treatments:
        treatment_name = "n_" + treatment.lower()
        ga_data_final[treatment_name] = np.zeros(len(ga_data_final))
        for col in tooth_cols:
            ga_data_final[treatment_name] += ga_data_final[col] == treatment
        ga_data_final["n_treated"] += ga_data_final[treatment_name]

    # change any non Pakistani / British categories to "other"
    ethnicity_cols = ['ethnicity', 'mother_ethnicity', 'father_birthplace',
                       'mother_birthplace', 'questionaire_language']
    england_pakistan_syn = [ "England", "Pakistan", "White British",
                             "Pakistani", "United Kingdom", "English"]

    for col in ethnicity_cols:
        cat_map = {val: "Other" if val not in england_pakistan_syn else val
                   for val in ga_data_final[col]}
        na_map = ga_data_final[col].isna()
        ga_data_final.loc[~na_map, col] = (ga_data_final
                                           .loc[~na_map, col]
                                           .map(cat_map))

    # drop unnecessary columns
    id_cols = ["ChildID", "PregnancyID", "MotherID", "FatherID"]
    if not include_ids:
        ga_data_final.drop(id_cols, axis=1, inplace=True)
        epi_data_final.drop(id_cols, axis=1, inplace=True)


    return ga_data_final, epi_data_final


def generate_tooth_level_data(df):
    
    # define tooth data and different treatment categories
    tooth_cols = [col for col in df.columns
                  if len(col) == 3 and col != "Epi"]
    treatments = ["Extracted", "Filled", "Sealed", "Crowned"]
    
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
    non_extractions =  ["n_sealed", "n_filled", "n_crowned"]
    tooth_df["n_non_extractions"] = tooth_df[non_extractions].sum(axis=1)
    
    return tooth_df


