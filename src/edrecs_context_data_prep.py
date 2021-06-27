
import pandas as pd
import numpy as np
import os
from ga_data_prep import *
from sklearn.model_selection import train_test_split


def clean_context_data(df):

    df = df.copy()

    # convert missing eal to "unknown"
    df["edcont_eal"] = df.edcont_eal.fillna(2)

    # convert missing ethnic origin to "other"
    df["edcont_ethnic_origin"] = df.edcont_ethnic_origin.fillna(10)

    # convert missing sen to mode = "not sen"
    df["edcont_sen"] = df.edcont_sen.fillna(2)

    # convert numeric columns to int
    for col in df.columns:
        if col == "entity_id":
            continue
        df[col] = df[col].astype("int16")

    return df


def convert_ethnicity_categories(df):
    df = df.copy()

    ethnicity_name = ("ethnicity" if "ethnicity" in df.columns
                      else "edcont_ethnic_origin")

    if df[ethnicity_name].dtype != np.dtype("int16"):
        raise ValueError("Categories must be numeric to convert")

    df[ethnicity_name] = df[ethnicity_name].apply(
        lambda x: x if x in [3, 8] else 10
    ).astype("int16")
    return df


def convert_sen_categories(df):

    df = df.copy()

    sen_name = "sen" if "sen" in df.columns else "edcont_sen"

    if df[sen_name].dtype != np.dtype("int16"):
        raise ValueError("Categories must be numeric to convert")

    df[sen_name] = df[sen_name].apply(
        lambda x: x if x == 2 else 3
    ).astype("int16")

    return df


def convert_context_numerical_categories(df):

    df = df.copy()

    # convert academicyear to full year
    df["edcont_academicyear"] = ((df.edcont_academicyear + 2010)
                                 .astype("category"))

    category_to_map_dict = {}
    # convert term of birth
    birth_term_codes = {
        1: "autumn",
        2: "spring",
        3: "summer"
    }
    category_to_map_dict["edcont_actermbirth"] = birth_term_codes

    # convert english additional language
    eal_codes = {
        1: "no",
        2: "unknown",
        3: "yes"
    }
    category_to_map_dict["edcont_eal"] = eal_codes

    # convert ethnic origin
    ethnic_origin_codes = {
        1: "bangladeshi",
        2: "indian",
        3: "pakistani",
        4: "other_asian",
        5: "african",
        6: "black_other",
        7: "mixed",
        8: "white_british",
        9: "white_other",
        10: "other",
    }
    category_to_map_dict["edcont_ethnic_origin"] = ethnic_origin_codes

    # convert yes / no features
    yn_codes = {
        1: "no",
        2: "yes"
    }
    category_to_map_dict["edcont_fsm"] = yn_codes
    category_to_map_dict["edcont_g_t"] = yn_codes
    category_to_map_dict["edcont_lac"] = yn_codes

    # convert gender
    gender_codes = {
        1: "female",
        2: "male"
    }
    category_to_map_dict["edcont_gender"] = gender_codes

    # convert special ed needs
    sen_codes = {
        1: "echp",
        2: "not_sen",
        3: "sen_support",
        4: "school_action",
        5: "school_action_plus",
        6: "statement"
    }
    category_to_map_dict["edcont_sen"] = sen_codes

    for category, map in category_to_map_dict.items():
        df[category] = (df[category]
                        .map(map)
                        .astype("category"))

    return df


def rename_context_cols(df):

    df = df.copy()
    rename_cols = {'edcont_academicyear': "year_started_school",
                   'edcont_actermbirth': "birthday_academic_term",
                   'edcont_eal': "eal",
                   'edcont_ethnic_origin': "ethnicity",
                   'edcont_fsm': "fsm",
                   'edcont_g_t': "gifted",
                   'edcont_gender': "gender",
                   'edcont_lac': "looked_after",
                   'edcont_sen': "sen"}

    df = df.rename(rename_cols, axis=1)

    return df


def add_has_dental_ga_to_context(context_df, ga_df):

    seq_name = "GASequence" if "GASequence" in ga_df.columns else "ga_sequence"

    # drop second GAs
    second_gas = ga_df[seq_name] == 2
    ga_data = ga_df[~second_gas]

    # merge requred ga data into context data
    ga_cols = ["entity_id", "has_dental_ga"]
    output_df = context_df.merge(ga_data[ga_cols],
                                 on="entity_id",
                                 how="left")
    output_df["has_dental_ga"] = output_df["has_dental_ga"].fillna(0)

    return output_df


def split_context_ga_data(context_ga_df):
    X_train, X_test, y_train, y_test = train_test_split(
        context_ga_df.drop("has_dental_ga", axis=1),
        context_ga_df.has_dental_ga,
        test_size=0.33,
        random_state=42,
        stratify=context_ga_df.has_dental_ga
    )
    train_df, test_df = X_train, X_test
    train_df["has_dental_ga"] = y_train
    test_df["has_dental_ga"] = y_test

    return train_df, test_df


def prep_context_ga_data(data_dir):

    ga_data_dir = os.path.join(data_dir, "bib_data/dental/dental_ga")
    ga_path = os.path.join(ga_data_dir, "data.csv")
    ga_df = pd.read_csv(ga_path)
    ga_df = ga_df.pipe(clean_ga_data)

    education_data_dir = os.path.join(data_dir, "bib_data/education")
    context_path = os.path.join(education_data_dir, "context/data.csv")
    context_df = pd.read_csv(context_path)
    context_df = context_df.pipe(clean_context_data)

    context_df = add_has_dental_ga_to_context(context_df, ga_df)
    train_df, test_df = split_context_ga_data(context_df)

    return train_df, test_df

