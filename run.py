# built-in imports
import logging
from pprint import pprint
import datetime

# internal imports
from mongo import MongoClient
import pymongo

# third-party inports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import linregress
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from preprocessing_functions import *


NAME = "Demo für Mohammad"

logging.basicConfig(
    level=logging.INFO,
    format=f"[%(asctime)s] {NAME} - %(levelname)s: %(message)s",
    filename="logging.log",
    filemode="w",
)

MONGO = MongoClient()


if __name__ == "__main__":

    MONGO.set_collection("mohammadDB", "wohnung_kaufen")

    # Match stages
    stage_match_AP_community_Munich = {"$match": {"AP_community": "München"}}
    stage_match_numberOfRooms_15 = {"$match": {"realEstate.numberOfRooms": 1.5}}
    stage_match_firstMayWeek_observation = {
        "$match": {
            "observationDate": {
                "$gte": "2022-05-01T00:00:00.000000",
                "$lt": "2022-05-08T00:00:00.000000",
            }
        }
    }
    stage_match_livingspace_gte100m2 = {
        "$match": {"realEstate.livingSpace": {"$gte": 100}}
    }
    stage_match_numberOfRooms_lte3 = {
        "$match": {"realEstate.numberOfRooms": {"$lte": 3}}
    }

    # Sort stages
    stage_sort_by_obseravtionDate = {"$sort": {"observationDate": pymongo.DESCENDING}}
    stage_sort_by_livingspace = {"$sort": {"Wohnfläche": pymongo.DESCENDING}}
    stage_sort_by_price = {"$sort": {"Kaufpreis": pymongo.DESCENDING}}

    # Limit documents
    stage_limit_documents = {"$limit": 20}

    # Project stages
    stage_project_example = {
        "$project": {
            "observationDate": "$observationDate",
            "AP_community": "$AP_community",
            "Anzahl_Räume": "$realEstate.numberOfRooms",
            "Kaufpreis": "$realEstate.price.value",
            "Wohnfläche": "$realEstate.livingSpace",
            "Einbauküche": "$realEstate.builtInKitchen",
            "Balkon": "$realEstate.balcony",
            "Garten": "$realEstate.garden",
            "Lift": "$realEstate.lift",
            "Gästetoilette": "$realEstate.guestToilet",
            "Keller": "$realEstate.cellar",
        }
    }

    # Fill the pipeline(s) with the necessary stages
    pipeline = [
        stage_project_example,
        # stage_sort_by_price,
    ]

    # Pipeline für alle Inserate aus München, die zwischen 1.5.22 und 7.5.22 in die DB gezogen wurden
    pipeline_MucMay = [
        stage_match_AP_community_Munich,
        stage_match_firstMayWeek_observation,
        stage_project_example,
        stage_sort_by_obseravtionDate,
    ]

    # Pipeline für die 20 größtflächigsten Inserate, die max über 3 Zimmer und über mindestens 100m2 an Wohnfläche verfügen
    pipeline_2room50m2 = [
        stage_match_livingspace_gte100m2,
        stage_match_numberOfRooms_lte3,
        stage_project_example,
        stage_sort_by_livingspace,
        stage_limit_documents,
    ]

    # Aggregation
    # cursor = MONGO.collection.aggregate(pipeline_MucMay)

    # Daten der Aggregation in Pandas Dataframe ziehen
    # df_raw = pd.DataFrame.from_records(list(cursor))
    # df_raw.to_csv("raw_data.csv", index = False)

    # Lesen des csv Files (schneller als die Daten immmer aus MongoDB zu ziehen)
    df_raw = pd.read_csv("data/raw_data.csv")
    # df_raw.sort_values(by=["Kaufpreis"])

    # for doc in cursor:
    # logging.info("Reading record with source_id %s", doc["source_id"])
    # pprint(doc)
    # break

    # andere outlier funktionen nutzen
    # Loop for removing the outliers
    cleaned_data = df_raw
    # print(cleaned_data.info())
    # Looped 5 times so no outlier remains
    for i in range(5):
        outlier_index_list = []
        for attribute in ["Kaufpreis", "Wohnfläche", "Anzahl_Räume"]:
            outlier_index_list.extend(outliers(cleaned_data, attribute))
        cleaned_data = remove_outliers(cleaned_data, outlier_index_list)
    # print(cleaned_data.info())

    # Data that is used for ML model
    # Schleife bauen
    cleaned_data["Einbauküche"] = cleaned_data["Einbauküche"].astype("category")
    cleaned_data["Einbauküche"] = cleaned_data["Einbauküche"].cat.codes

    cleaned_data["Balkon"] = cleaned_data["Balkon"].astype("category")
    cleaned_data["Balkon"] = cleaned_data["Balkon"].cat.codes

    cleaned_data["Garten"] = cleaned_data["Garten"].astype("category")
    cleaned_data["Garten"] = cleaned_data["Garten"].cat.codes

    cleaned_data["Lift"] = cleaned_data["Lift"].astype("category")
    cleaned_data["Lift"] = cleaned_data["Lift"].cat.codes

    cleaned_data["Gästetoilette"] = cleaned_data["Gästetoilette"].astype("category")
    cleaned_data["Gästetoilette"] = cleaned_data["Gästetoilette"].cat.codes

    cleaned_data["Keller"] = cleaned_data["Keller"].astype("category")
    cleaned_data["Keller"] = cleaned_data["Keller"].cat.codes

    numerical = cleaned_data.select_dtypes(include="float64").columns
    # Standardize the data by StandardScaling
    # cleaned_data.loc[:,numerical] = preprocessing.StandardScaler().fit_transform(cleaned_data.loc[:,numerical])

    # Normalize the data by MinMaxScaling
    # cleaned_data.loc[:,numerical] = preprocessing.MinMaxScaler().fit_transform(cleaned_data.loc[:,numerical])

    # Export the normalized cleaned Dataframe to csv file
    # cleaned_data.to_csv("data/normalized_cleaned_data.csv", index = False)

    # print(cleaned_data.isnull().sum())

    AP_communitys = cleaned_data["AP_community"].unique()

    normalized_cleaned_data = pd.read_csv("data/normalized_cleaned_data.csv")
    # print(normalized_cleaned_data.info())
    df_Muc = cleaned_data[cleaned_data["AP_community"] == "München"]
    # print(df_Muc.head())
    X = df_Muc.drop(["_id", "observationDate", "AP_community", "Kaufpreis"], axis=1)
    y = df_Muc["Kaufpreis"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    # Aufrufen der Funktionen
    """RandomForestRegression()
    GradientBoostingRegression()
    XGBoostRegression()
    CatBoostRegression()
    LightGBMRegression()"""
