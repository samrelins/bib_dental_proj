
import pandas as pd
import numpy as np
import os
from ga_data_prep import *
from sklearn.model_selection import train_test_split


def clean_context_data(df):

    df = df.copy()

    # convert academicyear to categories named with full year
    df["edcont_academicyear"] = ((df.edcont_academicyear + 2010)
                                 .astype("category"))

    # convert missing eal to "unknown" and name categories
    df["edcont_eal"] = df.edcont_eal.fillna(2)
    eal_codes = {
        1: "no",
        2: "unknown",
        3: "yes"
    }
    df["edcont_eal"] = (df.edcont_eal
                        .map(eal_codes)
                        .astype("category"))

    # convert ethnic origin to "white british"/ "pakistani" / "other"
    df["edcont_ethnic_origin"] = df["edcont_ethnic_origin"].apply(
        lambda x: x if x in [3, 8] else 10
    )
    ethnic_origin_codes = {
        3: "pakistani",
        8: "white_british",
        10: "other",
    }
    df["edcont_ethnic_origin"] = (df.edcont_ethnic_origin
                                  .map(ethnic_origin_codes)
                                  .astype("category"))

    # convert free school meals to binary "gifted" variable
    df["fsm"] = df["edcont_fsm"].apply(
        lambda x: True if x == 2 else False
    )

    # convert gifted / talented to binary "gifted" variable
    df["gifted"] = df["edcont_g_t"].apply(
        lambda x: True if x == 2 else False
    )

    # convert gender to binary "male" variable
    df["male"] = df["edcont_gender"].apply(
        lambda x: True if x == 2 else False
    )

    # convert looked after to binary  variable
    df["looked_after"] = df["edcont_lac"].apply(
        lambda x: True if x == 2 else False
    )

    # convert sen to binary "has sen" variable
    df["sen"] = df["edcont_sen"].apply(
        lambda x: False if x == 2 else True
    )

    # remove unneeded cols
    drop_cols = ['has_edrecs_context', 'edcont_actermbirth', 'edcont_fsm',
                 'edcont_g_t', 'edcont_gender', 'edcont_lac', 'edcont_sen',
                 'has_edcont']
    df.drop(drop_cols, axis=1, inplace=True)

    rename_cols = {'edcont_academicyear': "year_started_school",
                   'edcont_eal': "eal",
                   'edcont_ethnic_origin': "ethnicity"}
    df.rename(rename_cols, axis=1, inplace=True)

    return df


def clean_phonics_data(df):

    df = df.copy()

    # create "other" category in which to lump all pass / non-pass results incl. nas
    df["phonics_grade1"] = df.phonics_grade1.apply(
        lambda x: "expected" if x == 1 else "below_or_missing"
    ).astype("category")

    # fill mark nas with zero
    df.phonics_mark1.fillna(0, inplace=True)

    # convert phonics standard to binary variable
    df["meets_phonics_standard"] = df.phonics_standard1.apply(
        lambda x: True if x == 2 else False
    )

    # drop columns that aren't of interest
    # all test 2s are missing a significant number of obs
    drop_cols = ['has_edrecs_y1_phonics', 'has_edphon', 'phonics_acyrtested1',
                 'phonics_acyrtested2', 'phonics_grade2', 'phonics_mark2',
                 'phonics_standard1', 'phonics_standard2',
                 'phonics_testestablishment1', 'phonics_testestablishment2',
                 'phonics_yeartested1', 'phonics_yeartested2']
    df.drop(drop_cols, axis=1, inplace=True)

    return df


def clean_and_merge_ks1_data(bib_dir):

    # load pre 2016 ks1 data and drop unnecessary columns
    ks1_pre_2016_path = os.path.join(bib_dir, "education/ks1_pre_2016/data.csv")
    ks1_pre_2016_data = pd.read_csv(ks1_pre_2016_path)
    drop_cols_pre_2016 = ["has_edrecs_ks1_1", "has_edks11",
                          "ks1_pre2016_testestablishment", "ks1_pre2016_english",
                          "ks1_pre2016_acyrtested", "ks1_pre2016_yeartested"]
    ks1_pre_2016_data.drop(drop_cols_pre_2016, axis=1, inplace=True)

    # load post 2016 ks1 data and drop unnecessary columns
    ks1_post_2016_path = os.path.join(bib_dir, "education/ks1_post_2016/data.csv")
    ks1_post_2016_data = pd.read_csv(ks1_post_2016_path)
    drop_cols_post_2016 = ["has_edrecs_ks1_2", "has_edks12",
                           "ks1_post2016_testestablishment",
                           "ks1_post2016_acyrtested", "ks1_post2016_yeartested"]
    ks1_post_2016_data.drop(drop_cols_post_2016, axis=1, inplace=True)

    # convert to common column names
    def convert_ks1_colnames(name):
        split_name = name.split("_")
        return "ks1_" + split_name[-1] if split_name[0] == "ks1" else name

    ks1_pre_2016_data.columns = [convert_ks1_colnames(name)
                                 for name in ks1_pre_2016_data.columns]
    ks1_post_2016_data.columns = [convert_ks1_colnames(name)
                                  for name in ks1_post_2016_data.columns]

    # convert grades to approximate equivalents between pre / post 2016 tests
    pre_2016_scores_map = {
        1: "below_or_missing",
        2: "below_or_missing",
        3: "expected",
        4: "expected",
        5: "exceeding",
        6:"below_or_missing"
    }
    post_2016_scores_map = {
        1: "below_or_missing",
        2: "below_or_missing",
        3: "below_or_missing",
        4: "expected",
        5: "exceeding",
        6:"below_or_missing",
        7:"below_or_missing"
    }
    for subject in ["ks1_maths", "ks1_reading", "ks1_writing"]:
        ks1_pre_2016_data[subject] = (ks1_pre_2016_data[subject]
                                      .map(pre_2016_scores_map)
                                      .astype("category"))
        ks1_post_2016_data[subject] = (ks1_post_2016_data[subject]
                                       .map(post_2016_scores_map)
                                       .astype("category"))

    # append pre / post dataframes
    ks1_data = ks1_pre_2016_data.append(ks1_post_2016_data)

    # drop science grades - can't find a good equivalent between pre / post 2016 tests
    ks1_data.drop("ks1_science", axis=1, inplace=True)

    return ks1_data


def clean_eyfsp_data(df):

    df = df.copy()

    # remove unwanted columns
    drop_cols = ['has_edrecs_eyfsp_2', 'eyfsp_post2013_acyrtested',
                 'eyfsp_post2013_testestablishment', 'eyfsp_post2013_yeartested',
                 'has_eyfsp2']
    df.drop(drop_cols, axis=1, inplace=True)

    # convert eyfsp categories
    eyfsp_codes = {
        1: "below_or_missing",
        2: "expected",
        3: "exceeding",
        4: "below_or_missing",
    }
    for col in df.columns:
        if col == "entity_id":
            continue
        df[col] = (df[col]
                   .map(eyfsp_codes)
                   .astype("category"))

    rename_cols = {
        'eyfsp_post2013_com_elg01': "com_listening_attention",
        'eyfsp_post2013_com_elg02': "com_understanding",
        'eyfsp_post2013_com_elg03': "com_speaking",
        'eyfsp_post2013_phy_elg04': "phy_moving_handling",
        'eyfsp_post2013_phy_elg05': "phy_health_self_care",
        'eyfsp_post2013_pse_elg06': "pse_self_confidence_awareness",
        'eyfsp_post2013_pse_elg07': "pse_managing_feelings_behaviour",
        'eyfsp_post2013_pse_elg08': "pse_making_relationships",
        'eyfsp_post2013_lit_elg09': "lit_reading",
        'eyfsp_post2013_lit_elg10': "lit_writing",
        'eyfsp_post2013_mat_elg11': "mat_numbers",
        'eyfsp_post2013_mat_elg12': "mat_shapes_space_measures",
        'eyfsp_post2013_utw_elg13': "utw_people_communities",
        'eyfsp_post2013_utw_elg14': "utw_the_world",
        'eyfsp_post2013_utw_elg15': "utw_technology",
        'eyfsp_post2013_exp_elg16': "exp_using_media_materials",
        'eyfsp_post2013_exp_elg17': "exp_being_imaginative"}

    df.rename(rename_cols, axis=1, inplace=True)

    return df


def rename_context_cols(df):

    df = df.copy()

    df = df.rename(rename_cols, axis=1)

    return df


def return_merged_edrecs_df(bib_dir):
    # load eyfsp data
    eyfsp_path = os.path.join(bib_dir, "education/eyfsp_post_2013/data.csv")
    eyfsp_data = pd.read_csv(eyfsp_path)
    eyfsp_data = eyfsp_data.pipe(clean_eyfsp_data)

    # load context data
    context_path = os.path.join(bib_dir, "education/context/data.csv")
    context_data = pd.read_csv(context_path)
    context_data = context_data.pipe(clean_context_data)

    # load phonics data
    phonics_path = os.path.join(bib_dir, "education/y1_phonics/data.csv")
    phonics_data = pd.read_csv(phonics_path)
    phonics_data = phonics_data.pipe(clean_phonics_data)

    # load ks1 data
    ks1_data = clean_and_merge_ks1_data(bib_dir)

    # merge individual edrecs dataframes together
    dfs_to_merge = [context_data, phonics_data, ks1_data]
    edrecs_data = eyfsp_data.copy()
    for df in dfs_to_merge:
        edrecs_data = edrecs_data.merge(df, on="entity_id", how="inner")

    return edrecs_data


def return_merged_edrecs_ga_df(bib_dir, drop_comp_care_cases=False):
    edrecs_data = return_merged_edrecs_df(bib_dir)

    ga_path = os.path.join(bib_dir, "dental/dental_ga/data.csv")
    ga_data = pd.read_csv(ga_path)

    if drop_comp_care_cases:
        ga_data = (ga_data
                   .pipe(clean_ga_data)
                   .pipe(add_counts_of_treatments))
        comp_care_cases = ga_data.n_treated > 0
        ga_data = ga_data[~comp_care_cases][["entity_id", "has_dental_ga"]]
    else:
        ga_data = ga_data.pipe(clean_ga_data)[["entity_id", "has_dental_ga"]]

    ga_data.drop_duplicates(subset=["entity_id"], inplace=True)

    edrecs_ga_data = edrecs_data.merge(ga_data,
                                       on="entity_id",
                                       how="left")
    edrecs_ga_data["has_dental_ga"] = edrecs_ga_data.has_dental_ga.fillna(0)

    return edrecs_ga_data


