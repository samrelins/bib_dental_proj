import numpy as np
import os
import pandas as pd


def return_platoon_decay_data(bib_dir):

    # load tooth-level data
    teeth_path = os.path.join(bib_dir, "dental/plat_teeth/data.csv")
    teeth_data = pd.read_csv(teeth_path)


    # define tooth-level codes that dmft +1
    dmft_codes = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 0, 6: 1, 7: 0 }

    # separate primary and secondary tooth columns / features
    primary_teeth = [col for col in teeth_data.columns
                     if col[-2] in ["5", "6", "7", "8"]]
    permenent_teeth = [col for col in teeth_data.columns
                       if col[-2] in ["1", "2", "3", "4"]]

    # define new decay dataframe with same entities as tooth data
    decay_data = pd.DataFrame()
    decay_data["entity_id"] = teeth_data.entity_id

    # calculate dmft and primary caries experience for each entity
    decay_data["dmft"] = np.zeros(len(teeth_data))
    for tooth in primary_teeth:
        decay_data["dmft"] += teeth_data[tooth].map(dmft_codes)
    decay_data["primary_caries"] = decay_data.dmft > 0

    # calculate DMFT and secondary caries experience for each entity
    decay_data["DMFT"] = np.zeros(len(teeth_data))
    for tooth in permenent_teeth:
        decay_data["DMFT"] += teeth_data[tooth].map(dmft_codes)
    decay_data["secondary_caries"] = decay_data.DMFT > 0

    # add "group" feature from assessed data that details extraction groups
    ass_path = os.path.join(bib_dir, "dental/plat_assessed/data.csv")
    ass_data = pd.read_csv(ass_path)
    ass_data.rename({"BiBPersonID": "entity_id", "Group": "group"},
                    axis=1,
                    inplace=True)
    decay_data = decay_data.merge(ass_data[["entity_id", "group"]],
                                  on="entity_id",
                                  how="left")
    decay_data["group"] = decay_data.group.fillna(0)

    # define overall caries experience feature
    caries_in_obs = decay_data.dmft + decay_data.DMFT > 0
    has_had_extractions = decay_data.group > 0
    decay_data["caries_experience"] = ((caries_in_obs | has_had_extractions)
                                       .astype("int"))

    return decay_data

