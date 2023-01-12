# %%
# Imports
import sys
import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingRegressor

import os
import sys

module_path = os.path.abspath(
    os.path.join(
        r"C:\Users\soube\OneDrive\Desktop\Hammudi\Bachelorarbeit\Repository\AP-rent-determination\tree_models.py"
    )
)
if module_path not in sys.path:
    sys.path.append(module_path)

from tree_model_functions import *

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# Selecting the DataSource
dataSource = r"C:\Users\soube\OneDrive\Desktop\Hammudi\Bachelorarbeit\Repository\AP-rent-determination\students_data\cleaned_data_conf_with_IQR_removal.csv"

# Selecting columns to drop out of featureList and creating LabelList
featureDropList = [
    "_id",
    "observationDate",
    "city",
    "AP_community",
    "community_id",
    "base_rent",
    "qm2_rent",
    "DE_qm2_rent",
]
LabelList = ["qm2_rent"]

# Create DataFrame from DataSource
try:
    dataframe = import_data(dataSource)
except:
    dataframe = pd.read_csv(dataSource)


dataframe.drop(dataframe.filter(regex="second"), axis=1, inplace=True)
dataframe.drop(dataframe.filter(regex="third"), axis=1, inplace=True)

# %%
# Remove all hyphers from states
dataframe["state"] = (
    dataframe["state"].astype(str).apply(lambda x: x.replace("-", "")).astype(str)
)

# %%
for i in range(0, len(dataframe)):
    dataframe["postcode"][i] = str(dataframe["postcode"][i]).zfill(5)

# %%
# only keep first two digits of postcodes
for i in range(0, len(dataframe)):
    dataframe["postcode"][i] = dataframe["postcode"][i][:2]

# %%
# Create function to get first n digits of a number
import math


def first_n_digits(num, n):
    return num // 10 ** (int(math.log(num, 10)) - n + 1)


# %%
"""# only keep first two digits of postcodes
for i in range(0, len(dataframe)):
    dataframe["postcode"][i] = first_n_digits(dataframe["postcode"][i], 2)"""

# %%
# Rename states based on their postcodes
"""for i in range(0, 100):
    dataframe.loc[dataframe['postcode'] == i, 'state'] = str(i)"""

# %%
# Rename states based on their postcodes
for i in range(0, len(dataframe)):
    dataframe["state"][i] = dataframe["postcode"][i]

# %%
# Convert postcodes to int
dataframe["postcode"] = dataframe["postcode"].astype(str).astype(int)

# %%
# Creating test and trainset like this, that every state is represented 80/20 in these sets
# If train_test_split without looping throug the states first, not all states would have 80/20 representation

# Create list of unique states
states = dataframe["state"].unique()

# Create list for model scores
train_set = pd.DataFrame()
test_set = pd.DataFrame()

for state in states:
    df = dataframe
    df = df[df["state"] == state]

    # Create feature and label lists
    y = df[LabelList]
    X = df.drop(featureDropList, axis=1)
    feature_list = list(X.columns)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    trainset = pd.concat([X_train, y_train["qm2_rent"]], axis=1)
    testset = pd.concat([X_test, y_test["qm2_rent"]], axis=1)

    train_set = pd.concat([train_set, trainset], axis=0)
    test_set = pd.concat([test_set, testset], axis=0)

X_train_all = train_set.drop("qm2_rent", axis=1)
X_test_all = test_set.drop("qm2_rent", axis=1)
y_train_all = test_set[["qm2_rent", "state"]]
y_test_all = test_set[["qm2_rent", "state"]]

# %%
# Create many Combinations for each postcodegroup

states = sorted(states)

postcode_groups = [00, 10, 20, 30, 40, 50, 60, 70, 80, 90]
combination_sets = {}

for state in states:
    for x in postcode_groups:
        if first_n_digits(int(state), 2) == x:
            for i in range(1, int(state) + 1):
                if x - 10 < i:
                    print(states[x - 10 : i])
                    combination_sets[f"Combination_{states[i - 1]}"] = train_set[
                        train_set.state.isin(states[x - 10 : i])
                    ]
                    combination_sets[f"Combination_{states[i - 1]}"].name = [
                        f"Combination_{states[i - 1]}"
                    ]


# %%
# Creating dataframes with collapsible states and regions for training the models

# Osten
combination_sets["Osten_1"] = train_set[train_set.state.isin(["02", "03"])]
combination_sets["Osten_2"] = train_set[train_set.state.isin(["01", "03"])]
combination_sets["Osten_3"] = train_set[train_set.state.isin(["01", "09"])]
combination_sets["Osten_4"] = train_set[train_set.state.isin(["01", "04"])]
combination_sets["Osten_5"] = train_set[train_set.state.isin(["04", "09"])]
combination_sets["Osten_6"] = train_set[train_set.state.isin(["01", "04", "09"])]
combination_sets["Osten_7"] = train_set[train_set.state.isin(["01", "04", "08", "09"])]
combination_sets["Osten_8"] = train_set[train_set.state.isin(["04", "06"])]
combination_sets["Osten_9"] = train_set[train_set.state.isin(["04", "08", "09"])]
combination_sets["Osten_10"] = train_set[train_set.state.isin(["08", "09"])]
combination_sets["Osten_11"] = train_set[train_set.state.isin(["06", "07"])]
combination_sets["Osten_12"] = train_set[train_set.state.isin(["06", "07", "08"])]
combination_sets["Osten_13"] = train_set[
    train_set.state.isin(["06", "07", "08", "09", "04"])
]
combination_sets["Osten_14"] = train_set[train_set.state.isin(["14", "15"])]
combination_sets["Osten_15"] = train_set[train_set.state.isin(["14", "16"])]
combination_sets["Osten_16"] = train_set[train_set.state.isin(["15", "16"])]
combination_sets["Osten_17"] = train_set[train_set.state.isin(["14", "15", "16"])]
combination_sets["Osten_18"] = train_set[train_set.state.isin(["15", "03"])]
combination_sets["Osten_19"] = train_set[train_set.state.isin(["14", "15", "03"])]
combination_sets["Osten_20"] = train_set[train_set.state.isin(["17", "18"])]
combination_sets["Osten_21"] = train_set[train_set.state.isin(["17", "19"])]
combination_sets["Osten_22"] = train_set[train_set.state.isin(["18", "19"])]
combination_sets["Osten_23"] = train_set[train_set.state.isin(["16", "17", "18", "19"])]
combination_sets["Osten_24"] = train_set[train_set.state.isin(["19", "23"])]
combination_sets["Osten_25"] = train_set[train_set.state.isin(["19", "21"])]
combination_sets["Osten_26"] = train_set[train_set.state.isin(["19", "21", "23"])]
combination_sets["Osten_27"] = train_set[train_set.state.isin(["19", "21", "23", "29"])]
combination_sets["Osten_28"] = train_set[train_set.state.isin(["19", "29", "39"])]
combination_sets["Osten_29"] = train_set[train_set.state.isin(["14", "39"])]
combination_sets["Osten_30"] = train_set[train_set.state.isin(["14", "39", "06"])]
combination_sets["Osten_31"] = train_set[train_set.state.isin(["14", "39", "06", "04"])]

# Norden
combination_sets["Norden_1"] = train_set[train_set.state.isin(["24", "25"])]
combination_sets["Norden_2"] = train_set[train_set.state.isin(["23", "24"])]
combination_sets["Norden_3"] = train_set[train_set.state.isin(["23", "24", "25"])]
combination_sets["Norden_4"] = train_set[train_set.state.isin(["21", "27"])]
combination_sets["Norden_5"] = train_set[train_set.state.isin(["21", "29"])]
combination_sets["Norden_6"] = train_set[train_set.state.isin(["21", "27", "29"])]
combination_sets["Norden_7"] = train_set[train_set.state.isin(["27", "28"])]
combination_sets["Norden_8"] = train_set[train_set.state.isin(["27", "28", "26"])]
combination_sets["Norden_9"] = train_set[train_set.state.isin(["29", "28", "39"])]
combination_sets["Norden_10"] = train_set[train_set.state.isin(["26", "49"])]
combination_sets["Norden_11"] = train_set[train_set.state.isin(["26", "27", "49"])]
combination_sets["Norden_12"] = train_set[train_set.state.isin(["29", "30"])]

# Westen
combination_sets["Westen_1"] = train_set[train_set.state.isin(["48", "49"])]
combination_sets["Westen_2"] = train_set[train_set.state.isin(["44", "45"])]
combination_sets["Westen_3"] = train_set[train_set.state.isin(["45", "46"])]
combination_sets["Westen_4"] = train_set[train_set.state.isin(["46", "47"])]
combination_sets["Westen_5"] = train_set[train_set.state.isin(["45", "46", "47"])]
combination_sets["Westen_6"] = train_set[train_set.state.isin(["42", "44", "58"])]
combination_sets["Westen_7"] = train_set[train_set.state.isin(["44", "58", "59"])]
combination_sets["Westen_8"] = train_set[train_set.state.isin(["48", "59", "33"])]
combination_sets["Westen_9"] = train_set[train_set.state.isin(["41", "50", "52"])]
combination_sets["Westen_10"] = train_set[train_set.state.isin(["42", "51", "58"])]
combination_sets["Westen_11"] = train_set[train_set.state.isin(["50", "52", "53"])]
combination_sets["Westen_12"] = train_set[train_set.state.isin(["54", "55", "56"])]
combination_sets["Westen_13"] = train_set[train_set.state.isin(["53", "56"])]
combination_sets["Westen_14"] = train_set[train_set.state.isin(["55", "56"])]
combination_sets["Westen_15"] = train_set[train_set.state.isin(["55", "56", "65"])]
combination_sets["Westen_16"] = train_set[train_set.state.isin(["54", "66"])]
combination_sets["Westen_17"] = train_set[train_set.state.isin(["57", "35"])]
combination_sets["Westen_18"] = train_set[train_set.state.isin(["50", "51", "53"])]
combination_sets["Westen_19"] = train_set[train_set.state.isin(["57", "58", "59"])]
combination_sets["Westen_20"] = train_set[train_set.state.isin(["51", "58", "57"])]
combination_sets["Westen_21"] = train_set[train_set.state.isin(["59", "48", "33"])]
combination_sets["Westen_22"] = train_set[train_set.state.isin(["66", "67"])]
combination_sets["Westen_23"] = train_set[train_set.state.isin(["63", "64"])]
combination_sets["Westen_24"] = train_set[train_set.state.isin(["68", "69"])]
combination_sets["Westen_25"] = train_set[train_set.state.isin(["64", "68", "69"])]
combination_sets["Westen_26"] = train_set[train_set.state.isin(["61", "65"])]
combination_sets["Westen_27"] = train_set[train_set.state.isin(["61", "65", "35"])]
combination_sets["Westen_28"] = train_set[
    train_set.state.isin(["61", "63", "65", "35", "36"])
]
combination_sets["Westen_29"] = train_set[train_set.state.isin(["67", "55"])]
combination_sets["Westen_30"] = train_set[
    train_set.state.isin(["54", "55", "66", "67"])
]

# Süden
combination_sets["Süden_1"] = train_set[train_set.state.isin(["78", "79"])]
combination_sets["Süden_2"] = train_set[train_set.state.isin(["78", "79", "77"])]
combination_sets["Süden_3"] = train_set[train_set.state.isin(["75", "76"])]
combination_sets["Süden_4"] = train_set[train_set.state.isin(["72", "78"])]
combination_sets["Süden_5"] = train_set[train_set.state.isin(["73", "74"])]
combination_sets["Süden_6"] = train_set[train_set.state.isin(["74", "97"])]
combination_sets["Süden_7"] = train_set[train_set.state.isin(["73", "89"])]
combination_sets["Süden_8"] = train_set[train_set.state.isin(["72", "88"])]
combination_sets["Süden_9"] = train_set[train_set.state.isin(["72", "88", "89"])]
combination_sets["Süden_10"] = train_set[train_set.state.isin(["72", "73", "88", "89"])]
combination_sets["Süden_11"] = train_set[train_set.state.isin(["76", "68"])]
combination_sets["Süden_12"] = train_set[train_set.state.isin(["74", "69"])]
combination_sets["Süden_13"] = train_set[train_set.state.isin(["74", "75", "76"])]
combination_sets["Süden_14"] = train_set[train_set.state.isin(["88", "89"])]
combination_sets["Süden_15"] = train_set[train_set.state.isin(["87", "88", "89"])]
combination_sets["Süden_16"] = train_set[train_set.state.isin(["86", "87", "88", "89"])]
combination_sets["Süden_17"] = train_set[train_set.state.isin(["81", "82"])]
combination_sets["Süden_18"] = train_set[train_set.state.isin(["81", "82", "83"])]
combination_sets["Süden_19"] = train_set[train_set.state.isin(["83", "84"])]
combination_sets["Süden_20"] = train_set[train_set.state.isin(["85", "84"])]
combination_sets["Süden_21"] = train_set[train_set.state.isin(["83", "84", "85"])]
combination_sets["Süden_22"] = train_set[train_set.state.isin(["86", "89"])]
combination_sets["Süden_23"] = train_set[train_set.state.isin(["86", "89", "85", "84"])]
combination_sets["Süden_24"] = train_set[train_set.state.isin(["84", "94", "93"])]
combination_sets["Süden_25"] = train_set[train_set.state.isin(["93", "94"])]
combination_sets["Süden_26"] = train_set[train_set.state.isin(["92", "93"])]
combination_sets["Süden_27"] = train_set[train_set.state.isin(["95", "96"])]
combination_sets["Süden_28"] = train_set[train_set.state.isin(["07", "63", "36"])]
combination_sets["Süden_29"] = train_set[train_set.state.isin(["98", "99"])]
combination_sets["Süden_30"] = train_set[train_set.state.isin(["98", "99", "07"])]
combination_sets["Süden_31"] = train_set[train_set.state.isin(["96", "95", "07"])]
combination_sets["Süden_32"] = train_set[train_set.state.isin(["98", "99", "36"])]

# Zentrum
combination_sets["Zentrum_1"] = train_set[train_set.state.isin(["38", "39"])]
combination_sets["Zentrum_2"] = train_set[train_set.state.isin(["38", "39", "06"])]
combination_sets["Zentrum_3"] = train_set[train_set.state.isin(["30", "31", "38"])]
combination_sets["Zentrum_4"] = train_set[train_set.state.isin(["32", "33"])]
combination_sets["Zentrum_5"] = train_set[train_set.state.isin(["34", "35", "36"])]
combination_sets["Zentrum_6"] = train_set[train_set.state.isin(["34", "37"])]
combination_sets["Zentrum_7"] = train_set[train_set.state.isin(["33", "34", "37"])]
combination_sets["Zentrum_8"] = train_set[train_set.state.isin(["34", "36", "37"])]
combination_sets["Zentrum_9"] = train_set[
    train_set.state.isin(["34", "35", "36", "37"])
]


# %%
# Create names for every created dataset

input_dfs_list = combination_sets.items()
for key, value in input_dfs_list:
    combination_sets[key].name = [f"{key}"]

# %%
combination_sets_keys = []
combination_sets_values = []

for key, value in combination_sets.items():
    combination_sets_keys.append(key)
    combination_sets_values.append(value)

# %%
# Create Validation Test sets for every state

validation_sets = {}
for state in states:
    validation_sets[f"{state}_Validation_X_TestSet"] = X_test_all.loc[
        X_test_all["state"] == state
    ].drop("state", axis=1)
    validation_sets[f"{state}_Validation_y_TestSet"] = (
        y_test_all.loc[y_test_all["state"] == state]
        .drop("state", axis=1)["qm2_rent"]
        .tolist()
    )

# %%
# Create validation_test_set_lists (two with keys, two with values)

validation_y_TestSets_keys = []
validation_y_TestSets_values = []

validation_X_TestSets_keys = []
validation_X_TestSets_values = []

i = 0
for key, value in validation_sets.items():
    if i % 2:
        validation_y_TestSets_keys.append(key)
        validation_y_TestSets_values.append(value)
    else:
        validation_X_TestSets_keys.append(key)
        validation_X_TestSets_values.append(value)

    i = i + 1

# %%
# Create list of dataframes that are used for training the model

dataframes = {}
for state in states:
    dataframes[f"{state}"] = train_set.loc[train_set["state"] == state]

# %%
# Append all dfs from combination_sets to dataframes_list

for key, df in zip(combination_sets_keys, combination_sets_values):
    dataframes[key] = df

# %%
# Create dataframes_lists (one with keys, one with values)

dataframes_keys = []
dataframes_values = []

for key, value in dataframes.items():
    dataframes_keys.append(key)
    dataframes_values.append(value)

# %%
# Create dict with best_score and best_model_list for every state

best_scores = {}
best_models = {}
for state in states:
    best_scores[f"{state}_best_score"] = 0
    best_models[f"{state}_best_model"] = []

# Create best_score_lists (one with keys, one with values) (For later tracking of best score per state)

best_scores_keys = []
best_scores_values = []

for key, value in best_scores.items():
    best_scores_keys.append(key)
    best_scores_values.append(value)


# Create best_model_lists (one with keys, one with values) (For later tracking of best model per state)

best_models_keys = []
best_models_values = []

for key, value in best_models.items():
    best_models_keys.append(key)
    best_models_values.append(value)

# %%
# Loop throug all states and train them seperately

# Create list for most important features
feature_importances = []

# Create list for model scores
state_prediction_score = []

# Clear the scores_file
open(
    r"C:\Users\soube\OneDrive\Desktop\Hammudi\Bachelorarbeit\Repository\AP-rent-determination\tree_models\randomForest_scores.txt",
    "w",
).close()

for df in dataframes_values:
    df = df

    # Create feature and label lists
    y_train = df[LabelList]
    X_train = df.drop(["qm2_rent", "state"], axis=1)
    feature_list = list(X_train.columns)

    # y = np.array(y)
    # X = np.array(X)

    # Instantiate model
    rf = XGBRegressor(
        colsample_bytree=0.6,
        eta=0.1,
        gamma=0,
        max_depth=10,
        min_child_weight=1,
        n_estimators=100,
        random_state=0,
        reg_alpha=0.8,
        reg_lambda=1,
        subsample=0.6,
    )

    # Train the model on training data
    rf.fit(X_train, y_train.values.ravel())

    i = 0

    for X, y in zip(validation_X_TestSets_values, validation_y_TestSets_values):

        # Use the Regressors's predict method on the test data
        predictions = rf.predict(X)

        assert len(predictions) == len(y), "Length of predictions is not len y_test"
        # Calculate relative prediction errors
        errors = [
            100 * (abs(predictions[i] - y[i]) / y[i]) for i in range((len(predictions)))
        ]

        # Count of predictions that are at least 10% accurate
        count_good_predictions = sum(1 for i in errors if i <= 10)

        # Proportion of good predictions for the Testset
        good_predictions = round(
            np.mean(100 * (count_good_predictions / len(errors))), 2
        )

        state_prediction_score.append(
            [
                "Prediction on dataframe: ",
                df["state"].unique().tolist(),
                "Evaluating with Dataframe: ",
                states[i],
                good_predictions,
                df.shape[0],
            ]
        )

        # Compare performance of every state with every model to get best model for every state
        for state in states:
            if state == states[i]:
                if good_predictions > best_scores_values[i]:
                    best_scores_values[i] = good_predictions
                    best_models_values[i] = [
                        "Prediction on dataframe: ",
                        df["state"].unique().tolist(),
                        "Evaluating with Dataframe: ",
                        states[i],
                        "Prediction score on test data: ",
                        good_predictions,
                        "Number of rows of training data: ",
                        df.shape[0],
                    ]

        # Write all scores to a file
        with open(
            r"C:\Users\soube\OneDrive\Desktop\Hammudi\Bachelorarbeit\Repository\AP-rent-determination\tree_models\randomForest_scores.txt",
            "a",
        ) as f:

            f.write("The model got trained on:")
            f.write("\n")
            dataframe_name = repr(df["state"].unique())
            f.write(dataframe_name)
            f.write("\n")
            f.write("\n")
            f.write("The model got evaluated with:")
            f.write("\n")
            state_name = repr(states[i])
            f.write(state_name)
            f.write("\n")
            f.write("\n")
            f.write("Model score:")
            f.write("\n")
            good_predictions = repr(good_predictions)
            f.write(good_predictions)
            f.write("\n")
            f.write("\n")
            f.write("Train data shape:")
            f.write("\n")
            train_data_shape = repr(df.shape[0])
            f.write(train_data_shape)
            f.write("\n")
            f.write("\n")
            f.write("\n")
            f.write("\n")

        # Add 1 to get next state
        i = i + 1


# Calculate weighted overall model performance

model_performance = []

i = 0
for state in states:
    model_performance.append(
        (
            f"{state}_best_model",
            best_models_values[i],
            len(dataframe.loc[dataframe["state"] == state]),
        )
    )
    i += 1


# Convert state_prediction_score list into DataFrame
model_performance_df = pd.DataFrame(
    model_performance, columns=["model", "score", "inserates"]
)

# Weighted prediction score

number_of_inserates = model_performance_df["inserates"].sum()

model_performance_df["weighted_score"] = 0
for i in range(0, len(model_performance_df)):
    model_performance_df["weighted_score"][i] = (
        model_performance_df["score"][i][5] * model_performance_df["inserates"][i]
    )

final_prediction_score = (
    model_performance_df["weighted_score"].sum() / number_of_inserates
)


# Write best performing model for every state to file
with open(
    r"C:\Users\soube\OneDrive\Desktop\Hammudi\Bachelorarbeit\Repository\AP-rent-determination\tree_models\randomForest_best_scores.txt",
    "w",
) as f:

    f.write("Model performance:")
    f.write("\n")
    x = repr(final_prediction_score)
    f.write(x)
    f.write("\n")
    f.write("\n")
    f.write("Model with Hyperparameters:")
    f.write("\n")
    x = repr(rf)
    f.write(x)
    f.write("\n")
    f.write("\n")
    f.write("Features used:")
    f.write("\n")
    x = repr(feature_list)
    f.write(x)
    f.write("\n")
    f.write("\n")
    f.write("Dataframe:")
    f.write("\n")
    x = repr(dataSource)
    f.write(x)
    f.write("\n")
    f.write("\n")

    i = 0

    for state in states:
        f.write(f"{state}:")
        f.write("\n")

        for i in range(i, i + 1):
            best_model = repr(model_performance[i])
            f.write(best_model)
            f.write("\n")
            f.write("\n")

            i += 1

featureimportance_table = pd.DataFrame(
    {"Variable": X_train.columns, "Importance": rf.feature_importances_}
).sort_values("Importance", ascending=False)

# %%
from pprint import pprint

pprint(state_prediction_score)
