from ga_data_prep import *
import os
import pandas as pd


def return_baseline_df(bib_dir, select_cols=None):

    # load baseline data
    baseline_path = os.path.join(bib_dir, "other/base_m_survey.csv")
    baseline_data = pd.read_csv(baseline_path)
    if select_cols is not None:
        needed_cols = [col for col in ["entity_id", "BiBPregNumber"]
                       if col not in select_cols]
        baseline_data = baseline_data[select_cols + needed_cols]

    # load person data and extract relevant columns
    person_path = os.path.join(bib_dir, "other/person_info.csv")
    person_data = pd.read_csv(person_path)
    person_data = person_data[person_data.ParticipantType == "Child"]
    person_data = person_data[["BiBMotherID", "BiBPregNumber", "entity_id"]]

    # join baseline data with child IDs
    baseline_data.rename({"entity_id": "BiBMotherID"}, axis=1, inplace=True)
    output_df = person_data.merge(baseline_data,
                                  on=["BiBMotherID", "BiBPregNumber"],
                                  how="left")
    output_df.drop(["BiBMotherID", "BiBPregNumber"], axis=1, inplace=True)

    return output_df


def return_merged_baseline_ga_df(bib_dir, select_cols=None,
                                 drop_comp_care_cases=False):
    baseline_data = return_baseline_df(bib_dir, select_cols)

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

    baseline_ga_data = baseline_data.merge(ga_data,
                                           on="entity_id",
                                           how="outer")
    baseline_ga_data["has_dental_ga"] = baseline_ga_data.has_dental_ga.fillna(0)

    return baseline_ga_data


def return_reduced_baseline_ga_df(bib_dir):

    # load baseline / ga data points of interest
    cols_of_interest = ['entity_id', "BiBPregNumber", 'ben0nobenf', 'edu0fthede',
                        'edu0mumede', "eth0eth3gp", "fin0manfin",
                        'hhd0marchb', 'imd_2010_decile_nat', 'job0fthemp',
                        'job0mumemp', 'mbqlcasep5gp', "mms0mbkbmi", "smk0regsmk"]

    education_outcomes = {1: "<5_gcse", 2: "5_gcse", 3: "A_level",
                          4: "higher_than_A_level", 5: "other", 6: "dont_know",
                          7: "foreign_unknown"}

    rename_features = {
        "ben0nobenf": ("on_benefits", {1: "no",
                                       2: "yes"}),
        "edu0fthede": ("fathers_education", education_outcomes),
        "edu0mumede": ("mothers_education", education_outcomes),
        "eth0eth3gp": ("mothers_ethnicity", {1:"white_british",
                                             2: "pakistani",
                                             3: "other"}),
        "fin0manfin": ("managing_financially", {1: "living_comfortably",
                                                2: "doing_alright",
                                                3: "just_about_getting_by",
                                                4: "quite_difficult",
                                                5: "very_difficult",
                                                6: "no_answer"}),
        "hhd0marchb": ("married_cohabiting", {1: "married_n_cohabiting",
                                              2: "married_not_cohabiting",
                                              3: "not_cohabiting"}),
        "imd_2010_decile_nat": ("imd_decile", None),
        "job0fthemp": ("father_employment", {1: "non_manual",
                                             2: "manual",
                                             3: "self_employed",
                                             4: "student",
                                             5: "unemployed",
                                             6: "unknown"}),
        "job0mumemp": ("mother_employment", {1: "currently_employed",
                                             2: "previously_employed",
                                             3: "never_employed"}),
        "mbqlcasep5gp": ("socio_economic_pos", {1: "least_dep_most_edu",
                                                2: "employed_not_dep",
                                                3: "employed_dep",
                                                4: "benefits",
                                                5: "most_dep"}),
        "mms0mbkbmi": ("mothers_bmi", None),
        "smk0regsmk": ("mother_smoked", {1: "yes_over_year_ago",
                                         2: "yes_within_year",
                                         3: "yes",
                                         4: "no"})
    }


    baseline_ga = return_merged_baseline_ga_df(bib_dir,
                                               select_cols=cols_of_interest,
                                               drop_comp_care_cases = True)

    for feature in rename_features.keys():
        name, map = rename_features[feature]
        if map is not None:
            baseline_ga[name] = baseline_ga[feature].map(map).astype("category")
        else:
            baseline_ga[name] = baseline_ga[feature].astype("float")
        baseline_ga.drop(feature, axis=1, inplace=True)

    return baseline_ga

