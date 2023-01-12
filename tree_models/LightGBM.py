# %%
# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb


import os
import sys

module_path = os.path.abspath(
    os.path.join(
        r"C:\Users\soube\OneDrive\Desktop\Hammudi\Bachelorarbeit\Repository\AP-rent-determination\tree_models\tree_model_functions.py"
    )
)
if module_path not in sys.path:
    sys.path.append(module_path)

from tree_model_functions import *

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


# %%
# Reduce memory usage of dataframe
def reduce_mem_usage(df):
    """iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))

    return df


# Import dataframe from csv
def import_data(file):
    """create a dataframe and optimize its memory usage"""
    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)
    df = reduce_mem_usage(df)
    return df


# %%
# Selecting the DataSource
dataSource = r"C:\Users\soube\OneDrive\Desktop\Hammudi\Bachelorarbeit\Repository\AP-rent-determination\students_data\cleaned_data_with_IQR_removal.csv"

# Selecting columns to drop out of featureList and creating LabelList
featureDropList = [
    "_id",
    "observationDate",
    "state",
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

# Create list of unique states
states = dataframe["state"].unique()


# Create list for most important features
feature_importances = []

for category in ["postcode"]:
    dataframe[category] = dataframe[category].astype("category")
    dataframe[category] = dataframe[category].cat.codes

# dataframe.drop(dataframe.filter(regex = "second"), axis = 1, inplace = True)
# dataframe.drop(dataframe.filter(regex = "third"), axis = 1, inplace = True)

# %%
# Loop throug all states to train them seperately

# Create list for model scores
state_prediction_score = []

for state in states:
    df = dataframe
    df = df[df["state"] == state]

    # Create feature and label lists
    y = df[LabelList]
    X = df.drop(featureDropList, axis=1)
    feature_list = list(X.columns)

    y = np.array(y)
    X = np.array(X)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    # Instantiate model
    lgbm = lgb.LGBMRegressor()

    # Train the model on training data
    lgbm.fit(X_train, y_train)

    # Use the Regressors's predict method on the test data
    predictions = lgbm.predict(X_test)

    # Calculate the absolute errors
    errors = abs(predictions - y_test)

    # Print out the mean absolute error (mae)
    # print('Average model error:', round(np.mean(errors), 2), 'euros.')

    # Calculate relative prediction errors
    errors = [
        100 * (abs(predictions[i] - y_test[i]) / y_test[i])
        for i in range(min(len(predictions), len(y_test)))
    ]

    # Count of predictions that are at least 10% accurate
    count_good_predictions = sum(1 for i in errors if i <= 10)

    # Proportion of good predictions for the Testset
    good_predictions = round(np.mean(100 * (count_good_predictions / len(errors))), 2)
    # print('Percentage of predictions with less than 10 % deviation: ', good_predictions, '%.')

    state_prediction_score.append([state, good_predictions])

    # Get numerical feature importances
    importances = list(lgbm.feature_importances_)

    # List of tuples with variable and importance
    feature_importances_state = [
        (feature, round(importance, 3))
        for feature, importance in zip(feature_list, importances)
    ]
    feature_importances = feature_importances + feature_importances_state

# %%
print(state_prediction_score)

# %%
# Convert state_prediction_score list into DataFrame
state_prediction_df = pd.DataFrame(state_prediction_score, columns=["state", "score"])
# Calculate the average score of the model
prediction_score = state_prediction_df["score"].mean()
print("Model score :", prediction_score, "%")

# %% [markdown]
# ### Get rid of unnnecessary Variables ###

# %%
# Create Dataframe from feature importance list
feature_importances_df = pd.DataFrame(
    feature_importances, columns=["Variable", "Importance"]
)

feature_importances_df["average_Importance"] = (
    feature_importances_df["Importance"]
    .groupby(feature_importances_df["Variable"])
    .transform("mean")
)

# Get the mean Importances for every fetaure
mean_importances = feature_importances_df.groupby("Variable")["Importance"].mean()

# Create list for all mean_importances
cols = ["Variable", "average_Importance"]
mean_importances_list = feature_importances_df[cols].values.tolist()

# Sort and kick out all repetitive values
new_mean_importances_list = []
for i in mean_importances_list:
    if i not in new_mean_importances_list:
        new_mean_importances_list.append(i)
new_mean_importances_list = sorted(
    new_mean_importances_list, key=lambda x: x[1], reverse=True
)

# Importance sum
sum(x[1] for x in new_mean_importances_list)

# %%
# Reset style
plt.style.use("fivethirtyeight")

# list of x locations for plotting
x_values = list(range(len(mean_importances)))

# Make a bar chart
plt.bar(
    x_values,
    mean_importances,
    orientation="vertical",
    color="r",
    edgecolor="k",
    linewidth=1.2,
)

# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation="vertical")

# Axis labels and title
plt.ylabel("Importance")
plt.xlabel("Variable")
plt.title("Variable Importances")

# %%
# List of features sorted from most to least important
sorted_importances = [importance[1] for importance in new_mean_importances_list]
sorted_features = [importance[0] for importance in new_mean_importances_list]

# Cumulative importances
cumulative_importances = np.cumsum(sorted_importances)

# Make a line graph
plt.plot(x_values, cumulative_importances, "g-")

# Draw line at 90% of importance retained
plt.hlines(y=0.90, xmin=0, xmax=len(sorted_importances), color="r", linestyles="dashed")

# Format x ticks and labels
plt.xticks(x_values, sorted_features, rotation="vertical")

# Axis labels and title
plt.xlabel("Variable")
plt.ylabel("Cumulative Importance")
plt.title("Cumulative Importances")

# %%
# Find number of features for cumulative importance of 90%
# Add 1 because Python is zero-indexed
num_of_important_features = np.where(cumulative_importances > 2600)[0][0] + 1
print("Number of features for 90% importance:", num_of_important_features)

# %%
# List with most important features
new_mean_importances_list = new_mean_importances_list[:num_of_important_features]

# List with most important features without importances
new_mean_importances_list_names = [item[0] for item in new_mean_importances_list]

# %%
# Loop throug all states to train them seperately

# Create list for model scores
state_prediction_score_imp = []

for state in states:
    df = dataframe
    df = df[df["state"] == state]

    # Create feature and label lists
    y = df[LabelList]
    X = df[new_mean_importances_list_names]

    y = np.array(y)
    X = np.array(X)

    # Train test split
    important_X_train, important_X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    # Instantiate model
    lgbm = lgb.LGBMRegressor()

    # Train the model on training data
    lgbm.fit(important_X_train, y_train)

    # Use the Regressors's predict method on the test data
    predictions = lgbm.predict(important_X_test)

    # Calculate the absolute errors
    errors = abs(predictions - y_test)

    # Print out the mean absolute error (mae)
    # print('Average model error:', round(np.mean(errors), 2), 'euros.')

    # Calculate relative prediction errors
    errors = [
        100 * (abs(predictions[i] - y_test[i]) / y_test[i])
        for i in range(min(len(predictions), len(y_test)))
    ]

    # Count of predictions that are at least 10% accurate
    count_good_predictions = sum(1 for i in errors if i <= 10)

    # Proportion of good predictions for the Testset
    good_predictions = round(np.mean(100 * (count_good_predictions / len(errors))), 2)
    # print('Percentage of predictions with less than 10 % deviation: ', good_predictions, '%.')

    state_prediction_score_imp.append([state, good_predictions])

# %%
print(state_prediction_score_imp)

# %%
# Convert state_prediction_score list into DataFrame
state_prediction_df = pd.DataFrame(
    state_prediction_score_imp, columns=["state", "score"]
)
# Calculate the average score of the model
prediction_score_imp = state_prediction_df["score"].mean()
print("Model score :", prediction_score_imp, "%")

# %%
with open(
    r"C:\Users\soube\OneDrive\Desktop\Hammudi\Bachelorarbeit\Repository\AP-rent-determination\tree_models_notes\lgbm_scores.txt",
    "w",
) as f:
    f.write("Hyperparameters:")
    f.write("\n")
    params = repr(lgbm.get_params())
    f.write(params)
    f.write("\n")
    f.write("\n")
    f.write("Model score with all features:")
    f.write("\n")
    state_prediction_score = repr(state_prediction_score)
    f.write(state_prediction_score)
    f.write("\n")
    prediction_score = repr(prediction_score)
    f.write(prediction_score)
    f.write("\n")
    f.write("\n")
    f.write("Model score with important features:")
    f.write("\n")
    state_prediction_score = repr(state_prediction_score_imp)
    f.write(state_prediction_score)
    f.write("\n")
    prediction_score = repr(prediction_score_imp)
    f.write(prediction_score)
