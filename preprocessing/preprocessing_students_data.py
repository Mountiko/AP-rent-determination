# %%
# third-party imports
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import preprocessing

sys.path.insert(
    1,
    r"C:\Users\soube\OneDrive\Desktop\Hammudi\Bachelorarbeit\Repository\AP-rent-determination\preprocessing\preprocessing_functions.py",
)

# %%
dataSource = r"C:\Users\soube\OneDrive\Desktop\Hammudi\Bachelorarbeit\Repository\AP-rent-determination\wohnung_kaufen\raw_data.csv"
outlierList = ["livingSpace", "energyConsumption", "base_rent"]
categoryList = [
    "apartmentType",
    "condition",
    "energyEfficiencyClass",
    "energyType",
    "heatingType",
    "floorType",
]

# %%
properties = pd.read_csv(dataSource)

# %%
pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)

# %%
properties.dtypes

# %%
properties.describe()

# %%
properties.isna().sum()

# %%
import missingno as msno

# Plot correlation heatmap of missingness
msno.matrix(properties.iloc[:, :26])

# %%
msno.heatmap(properties)

# %%
# strong correlation between energyConsumption and energyEfficiencyClass. Confirm this by sorting either of the columns

msno.matrix(properties.iloc[:, :26].sort_values("energyConsumption"))

# The plot shows that if a data point is missing in energyConsumption, we can guess that it is also missing from energyEfficiencyClass column or vice versa.
# Because of this connection, we can safely say the missing data in both columns are not missing at random (MNAR).

# also a weak correlation between energyEfficiencyClass, energyType and constructionYear, which would indicate that energyType and constructionYear were not missing completely at random (MCAR) but has some relationship with missing values in energyEfficiencyClass. In other words, it is missing at random (MAR).


# %% [markdown]
# ## Check every column seperately ##

# %% [markdown]
# ### At first: Categorical Variables that need to be encoded ###

# %%
properties.head()

# %%
properties["parkingCount"] = properties["parkingCount"].replace(np.nan, 0)
properties["parkingCount"][properties["parkingCount"] > 0] = 1
properties.rename({"parkingCount": "parking"}, axis=1, inplace=True)

# %%
# Parse and change dataType
properties["energyConsumption"] = properties["energyConsumption"].str[:-11]
properties["energyConsumption"] = properties["energyConsumption"].astype(float)

# %%
# Sorted list of energyEfficiencyClass elements
sorted(properties["energyEfficiencyClass"].dropna().unique(), key=lambda g: g + ",")

# %%
# Encode energyEfficiencyClass
properties["energyEfficiencyClass"] = properties["energyEfficiencyClass"].replace(
    ["A+"], "1"
)
properties["energyEfficiencyClass"] = properties["energyEfficiencyClass"].replace(
    ["A"], "2"
)
properties["energyEfficiencyClass"] = properties["energyEfficiencyClass"].replace(
    ["B"], "3"
)
properties["energyEfficiencyClass"] = properties["energyEfficiencyClass"].replace(
    ["C"], "4"
)
properties["energyEfficiencyClass"] = properties["energyEfficiencyClass"].replace(
    ["D"], "5"
)
properties["energyEfficiencyClass"] = properties["energyEfficiencyClass"].replace(
    ["E"], "6"
)
properties["energyEfficiencyClass"] = properties["energyEfficiencyClass"].replace(
    ["F"], "7"
)
properties["energyEfficiencyClass"] = properties["energyEfficiencyClass"].replace(
    ["G"], "8"
)
properties["energyEfficiencyClass"] = properties["energyEfficiencyClass"].replace(
    ["H"], "9"
)
properties["energyEfficiencyClass"] = properties["energyEfficiencyClass"].replace(
    np.nan, -1
)

# %%
# One-Hot Encoding
apartmentTypeEncoded = pd.get_dummies(properties.apartmentType)
conditionEncoded = pd.get_dummies(properties.condition)
heatingTypeEncoded = pd.get_dummies(properties.heatingType)

# floorType already has value "Sonstiges", so all NaN get converted to Sonstiges
properties["floorType"] = properties["floorType"].replace(np.nan, "Sonstiges")
floorTypeEncoded = pd.get_dummies(properties.floorType)

# %%
properties = pd.concat(
    [
        properties,
        apartmentTypeEncoded,
        conditionEncoded,
        heatingTypeEncoded,
        floorTypeEncoded,
    ],
    axis=1,
)
properties.shape

# %%
# Only execute this code if OneHot-Encoding and Concat is done

del properties["apartmentType"]
del properties["condition"]
del properties["heatingType"]
del properties["floorType"]

# %%
pd.Series("".join(list(properties["energyType"].dropna().unique())).split(",")).unique()

# %%
# Boolean variables get 0's and 1's
properties.replace({False: 0, True: 1}, inplace=True)

# %% [markdown]
# ##### Clean Construction Year #####

# %%
# Show the biggest values to see false data
properties["constructionYear"].nlargest(n=10)
# Correct false handmade faults
properties.loc[properties["constructionYear"] == 9170, "constructionYear"] = 1970
properties.loc[properties["constructionYear"] == 2978, "constructionYear"] = 1978
properties.loc[properties["constructionYear"] == 2975, "constructionYear"] = 1975
properties.loc[properties["constructionYear"] == 5991.0, "constructionYear"] = 1991
properties.loc[properties["constructionYear"] == 2257.0, "constructionYear"] = 1957
# Drop all properties that finish in the future
properties.drop(properties[properties["constructionYear"] > 2022].index, inplace=True)

# %%
# Show the smallest values to see false data
properties["constructionYear"].nsmallest(n=5000)
# Drop all properties before 1900 because risk of handmade faults are too high
properties.drop(properties[properties["constructionYear"] < 1900].index, inplace=True)

# %% [markdown]
# ##### Clean roomCount #####

# %%
# Show the biggest values to see false data
properties["roomCount"].nlargest(n=20)
# Drop
properties.drop(properties[properties["roomCount"] > 9.5].index, inplace=True)

# %%
# Show the smallest values to see false dta
properties["roomCount"].nsmallest(n=10)

# %% [markdown]
# ##### Clean energyConsumption #####

# %%
# Show the biggest values to see false data
properties["energyConsumption"].nlargest(n=1000)
properties.drop(properties[properties["energyConsumption"] > 800].index, inplace=True)

# %%
properties["energyConsumption"].nsmallest(n=90000)
# Drop
properties.drop(properties[properties["energyConsumption"] < 20].index, inplace=True)

# Es ist trotzdem noch möglich dass erheblich Eingabefehler geacht wurden, zB 20.0 statt 200
# Dieses Feature lieber rausschmeißen (siehe später bei bins: energyConsumption steigt nicht an mit steigender Wohnfläche) -> fehlerhafte Daten

# %%
properties["energyConsumption"].nsmallest(n=90000)

# %% [markdown]
# ##### Clean livingSpace #####

# %%
# Show the biggest values to see false data
properties["livingSpace"].nlargest(n=1000)
# Drop
properties.drop(properties[properties["livingSpace"] > 350].index, inplace=True)

# %%
properties["livingSpace"].nsmallest(n=20000)
properties.drop(properties[properties["livingSpace"] < 10].index, inplace=True)

# %% [markdown]
# ##### Clean base_rent #####

# %%
# Show the biggest values to see false data
properties["base_rent"].nlargest(n=1000)

# %%
properties["base_rent"].nsmallest(n=1000)
properties.drop(properties[properties["base_rent"] < 100].index, inplace=True)

# %% [markdown]
# ### Fill out the rest NaN values ###

# %%
cleaned_data = properties

# %%
cleaned_data.isna().sum()

# %%
cleaned_data["floor"] = cleaned_data["floor"].replace(
    np.nan, cleaned_data["floor"].median()
)
cleaned_data["constructionYear"] = cleaned_data["constructionYear"].replace(
    np.nan, cleaned_data["constructionYear"].median()
)
cleaned_data["coord_confidence"] = cleaned_data["coord_confidence"].replace(np.nan, 0)

# %%
bin40qm = cleaned_data.loc[
    cleaned_data["livingSpace"] <= 40, "energyConsumption"
].median()
bin80qm = cleaned_data.loc[
    (cleaned_data["livingSpace"] > 40) & (cleaned_data["livingSpace"] <= 80),
    "energyConsumption",
].median()
bin120qm = cleaned_data.loc[
    (cleaned_data["livingSpace"] > 80) & (cleaned_data["livingSpace"] <= 120),
    "energyConsumption",
].median()
bin160qm = cleaned_data.loc[
    (cleaned_data["livingSpace"] > 120) & (cleaned_data["livingSpace"] <= 160),
    "energyConsumption",
].median()
bin200qm = cleaned_data.loc[
    (cleaned_data["livingSpace"] > 160) & (cleaned_data["livingSpace"] <= 200),
    "energyConsumption",
].median()
bin240qm = cleaned_data.loc[
    (cleaned_data["livingSpace"] > 200) & (cleaned_data["livingSpace"] <= 240),
    "energyConsumption",
].median()
bin280qm = cleaned_data.loc[
    (cleaned_data["livingSpace"] > 240) & (cleaned_data["livingSpace"] <= 280),
    "energyConsumption",
].median()
bin320qm = cleaned_data.loc[
    (cleaned_data["livingSpace"] > 280) & (cleaned_data["livingSpace"] <= 320),
    "energyConsumption",
].median()
bin360qm = cleaned_data.loc[
    (cleaned_data["livingSpace"] > 320) & (cleaned_data["livingSpace"] <= 360),
    "energyConsumption",
].median()

# %%
bins = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360]
labels = [
    "40qm2",
    "80qm2",
    "120qm2",
    "160qm2",
    "200qm2",
    "240qm2",
    "280qm2",
    "320qm2",
    "360qm2",
]
cleaned_data["energyConsumption_bin"] = pd.cut(
    cleaned_data["livingSpace"], bins=bins, labels=labels
)

cleaned_data["energyConsumption_bin"] = cleaned_data["energyConsumption_bin"].replace(
    ["40qm2"], bin40qm
)
cleaned_data["energyConsumption_bin"] = cleaned_data["energyConsumption_bin"].replace(
    ["80qm2"], bin80qm
)
cleaned_data["energyConsumption_bin"] = cleaned_data["energyConsumption_bin"].replace(
    ["120qm2"], bin120qm
)
cleaned_data["energyConsumption_bin"] = cleaned_data["energyConsumption_bin"].replace(
    ["160qm2"], bin160qm
)
cleaned_data["energyConsumption_bin"] = cleaned_data["energyConsumption_bin"].replace(
    ["200qm2"], bin200qm
)
cleaned_data["energyConsumption_bin"] = cleaned_data["energyConsumption_bin"].replace(
    ["240qm2"], bin240qm
)
cleaned_data["energyConsumption_bin"] = cleaned_data["energyConsumption_bin"].replace(
    ["280qm2"], bin280qm
)
cleaned_data["energyConsumption_bin"] = cleaned_data["energyConsumption_bin"].replace(
    ["320qm2"], bin320qm
)
cleaned_data["energyConsumption_bin"] = cleaned_data["energyConsumption_bin"].replace(
    ["360qm2"], bin360qm
)

# %%
bin360qm

# %%
cleaned_data.head()

# %%
# Fill out NaN values
cleaned_data["energyConsumption"] = np.where(
    cleaned_data.energyConsumption.isnull(),
    cleaned_data.energyConsumption_bin,
    cleaned_data.energyConsumption,
)

# %%
del cleaned_data["energyConsumption_bin"]

# %%
cleaned_data.shape

# %%
with open(
    r"C:\Users\soube\OneDrive\Desktop\Hammudi\Bachelorarbeit\Repository\AP-rent-determination\students_data\Numerical Feature Ranges_before_IQR_removal.txt",
    "w",
) as f:
    f.write("Numerical Feature Ranges BEFORE IQR removal:")
    f.write("\n")
    f.write("\n")
    livingSpace_max = repr(cleaned_data["livingSpace"].max())
    f.write("LivingSpace Max: " + livingSpace_max)
    f.write("\n")
    livingSpace_min = repr(cleaned_data["livingSpace"].min())
    f.write("LivingSpace Min: " + livingSpace_min)
    f.write("\n")
    f.write("\n")
    energyConsumption_max = repr(cleaned_data["energyConsumption"].max())
    f.write("energyConsumption Max: " + energyConsumption_max)
    f.write("\n")
    energyConsumption_min = repr(cleaned_data["energyConsumption"].min())
    f.write("energyConsumption Min: " + energyConsumption_min)
    f.write("\n")
    f.write("\n")
    base_rent_max = repr(cleaned_data["base_rent"].max())
    f.write("baseRent Max: " + base_rent_max)
    f.write("\n")
    base_rent_min = repr(cleaned_data["base_rent"].min())
    f.write("baseRent Min: " + base_rent_min)
    f.write("\n")
    f.write("\n")
    constructionYear_max = repr(cleaned_data["constructionYear"].max())
    f.write("constructionYear Max: " + constructionYear_max)
    f.write("\n")
    constructionYear_min = repr(cleaned_data["constructionYear"].min())
    f.write("constructionYear Min: " + constructionYear_min)
    f.write("\n")
    f.write("\n")
    roomCount_max = repr(cleaned_data["roomCount"].max())
    f.write("roomCount Max: " + roomCount_max)
    f.write("\n")
    roomCount_min = repr(cleaned_data["roomCount"].min())
    f.write("roomCount Min: " + roomCount_min)
    f.write("\n")
    f.write("\n")
    floor_max = repr(cleaned_data["floor"].max())
    f.write("floor Max: " + floor_max)
    f.write("\n")
    floor_min = repr(cleaned_data["floor"].min())
    f.write("floor Min: " + floor_min)
    f.write("\n")
    f.write("\n")
    data_shape = repr(cleaned_data.shape)
    f.write("Data Shape: " + data_shape)

# %%
# Detect outliers
def outliers(df, ft):
    Q1 = df[ft].quantile(0.25)
    Q3 = df[ft].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    ls = df.index[(df[ft] < lower_bound) | (df[ft] > upper_bound)]

    return ls


# Drop the detected outliers out of the Dataframe
def remove_outliers(df, ls):
    ls = sorted(set(ls))
    df = df.drop(ls)
    return df


# %%
cleaned_data = properties
for i in range(1):
    outlier_index_list = []
    for attribute in outlierList:
        outlier_index_list.extend(outliers(cleaned_data, attribute))
    cleaned_data = remove_outliers(cleaned_data, outlier_index_list)

# %%


with open(
    r"C:\Users\soube\OneDrive\Desktop\Hammudi\Bachelorarbeit\Repository\AP-rent-determination\students_data\Numerical Feature Ranges_after_IQR_removal.txt",
    "w",
) as f:
    f.write("Numerical Feature Ranges AFTER IQR removal:")
    f.write("\n")
    f.write("\n")
    livingSpace_max = repr(cleaned_data["livingSpace"].max())
    f.write("LivingSpace Max: " + livingSpace_max)
    f.write("\n")
    livingSpace_min = repr(cleaned_data["livingSpace"].min())
    f.write("LivingSpace Min: " + livingSpace_min)
    f.write("\n")
    f.write("\n")
    energyConsumption_max = repr(cleaned_data["energyConsumption"].max())
    f.write("energyConsumption Max: " + energyConsumption_max)
    f.write("\n")
    energyConsumption_min = repr(cleaned_data["energyConsumption"].min())
    f.write("energyConsumption Min: " + energyConsumption_min)
    f.write("\n")
    f.write("\n")
    base_rent_max = repr(cleaned_data["base_rent"].max())
    f.write("baseRent Max: " + base_rent_max)
    f.write("\n")
    base_rent_min = repr(cleaned_data["base_rent"].min())
    f.write("baseRent Min: " + base_rent_min)
    f.write("\n")
    f.write("\n")
    constructionYear_max = repr(cleaned_data["constructionYear"].max())
    f.write("constructionYear Max: " + constructionYear_max)
    f.write("\n")
    constructionYear_min = repr(cleaned_data["constructionYear"].min())
    f.write("constructionYear Min: " + constructionYear_min)
    f.write("\n")
    f.write("\n")
    roomCount_max = repr(cleaned_data["roomCount"].max())
    f.write("roomCount Max: " + roomCount_max)
    f.write("\n")
    roomCount_min = repr(cleaned_data["roomCount"].min())
    f.write("roomCount Min: " + roomCount_min)
    f.write("\n")
    f.write("\n")
    floor_max = repr(cleaned_data["floor"].max())
    f.write("floor Max: " + floor_max)
    f.write("\n")
    floor_min = repr(cleaned_data["floor"].min())
    f.write("floor Min: " + floor_min)
    f.write("\n")
    f.write("\n")
    data_shape = repr(cleaned_data.shape)
    f.write("Data Shape: " + data_shape)


cleaned_data.shape

# %%
# Code for later use maybe

# Removing outliers by IQR
"""cleaned_data = properties
for i in range(1):
    outlier_index_list = []    
    for attribute in outlierList:
        outlier_index_list.extend(outliers(cleaned_data, attribute))
    cleaned_data = remove_outliers(cleaned_data, outlier_index_list)
cleaned_data.shape"""

# Changing datatypes
"""for col in cleaned_data.columns:
    if cleaned_data[col].dtype == np.int8:
        cleaned_data[col] = cleaned_data[col].astype(float)
    if cleaned_data[col].dtype == np.int64:
        cleaned_data[col] = cleaned_data[col].astype(float)"""

# Changing dtype from float64 to float32
"""for dataset in datasets:
    # Select columns with 'float64' dtype  
    float64_cols = list(dataset.select_dtypes(include='float64'))

    # The same code again calling the columns
    dataset[float64_cols] = dataset[float64_cols].astype('float32')"""

# Normalize the data
"""for dataset in datasets:
    numerical = dataset.select_dtypes(include='float64').columns
    #Normalize the data by MinMaxScaling
    dataset.loc[:,numerical] = preprocessing.MinMaxScaler().fit_transform(dataset.loc[:,numerical])"""

# %%
# ["apartmentType", "condition", "heatingType", "floorType"] später durch one hot ersetzen
# energy type mit true false und neuen spalten ersetzen

for category in ["energyType"]:
    cleaned_data[category] = cleaned_data[category].astype("category")
    cleaned_data[category] = cleaned_data[category].cat.codes

# %%
cleaned_data.shape

# %%
cleaned_data["qm2_rent"] = cleaned_data["base_rent"] / cleaned_data["livingSpace"]

# %%


# %%
# Deutschlandweiter Quadratmeterpreis
cleaned_data["DE_qm2_rent"] = cleaned_data["qm2_rent"].median()

# %%
# Export data with conf 9 to csv file
cleaned_data_conf = cleaned_data[cleaned_data["coord_confidence"] == 9]
cleaned_data_conf.to_csv("cleaned_data_conf.csv", index=False)

# Export the normalized cleaned Dataframe to csv file
cleaned_data.to_csv("cleaned_data.csv", index=False)

# %%
# Deutschlandweiter Quadratmeterpreis
cleaned_data["DE_qm2_rent"] = cleaned_data["qm2_rent"].median()

# %%
datasets = [cleaned_data]

for dataset in datasets:
    dataset["averageQmRent_APC"] = (
        dataset["qm2_rent"].groupby(dataset["AP_community"]).transform("median")
    )
    dataset["averageQmRent_postcode"] = (
        dataset["qm2_rent"].groupby(dataset["postcode"]).transform("median")
    )

# %%
for dataset in datasets:
    cleaned_data["Abschlag_Faktor_APC"] = cleaned_data["qm2_rent"] / (
        cleaned_data["averageQmRent_APC"].astype(float)
        / cleaned_data["DE_qm2_rent"].astype(float)
    )
    cleaned_data["Abschlag_Faktor_postcode"] = cleaned_data["qm2_rent"] / (
        cleaned_data["averageQmRent_postcode"].astype(float)
        / cleaned_data["DE_qm2_rent"].astype(float)
    )
    cleaned_data["Absoluter_Abschlag_APC"] = cleaned_data["qm2_rent"] + (
        cleaned_data["averageQmRent_APC"].astype(float)
        - cleaned_data["DE_qm2_rent"].astype(float)
    )
    cleaned_data["Absoluter_Abschlag_postcode"] = cleaned_data["qm2_rent"] + (
        cleaned_data["averageQmRent_postcode"].astype(float)
        - cleaned_data["DE_qm2_rent"].astype(float)
    )

# %%
AP_community_Durchschnittspreise = cleaned_data[
    [
        "AP_community",
        "community_id",
        "postcode",
        "DE_qm2_rent",
        "averageQmRent_APC",
        "averageQmRent_postcode",
        "Abschlag_Faktor_APC",
        "Abschlag_Faktor_postcode",
        "Absoluter_Abschlag_APC",
        "Absoluter_Abschlag_postcode",
    ]
]

# %%
AP_community_Durchschnittspreise = AP_community_Durchschnittspreise.drop_duplicates()

# %%
del cleaned_data["averageQmRent_APC"]
del cleaned_data["averageQmRent_postcode"]
del cleaned_data["Abschlag_Faktor_APC"]
del cleaned_data["Abschlag_Faktor_postcode"]
del cleaned_data["Absoluter_Abschlag_APC"]
del cleaned_data["Absoluter_Abschlag_postcode"]
del cleaned_data["DE_qm2_rent"]

# %%
AP_community_Durchschnittspreise.to_csv("Abschlagliste_AP_communitys.csv", index=False)
