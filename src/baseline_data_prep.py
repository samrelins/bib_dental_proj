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
    baseline_data = baseline_data.merge(person_data,
                                    on=["BiBMotherID", "BiBPregNumber"],
                                    how="left")
    baseline_data.drop(["BiBMotherID", "BiBPregNumber"], axis=1, inplace=True)
    baseline_data.dropna(subset=["entity_id"], inplace=True)

    if select_cols is not None:
        baseline_data.dropna(inplace=True)

    return baseline_data


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
                                           how="left")
    baseline_ga_data["has_dental_ga"] = baseline_ga_data.has_dental_ga.fillna(0)

    return baseline_ga_data

