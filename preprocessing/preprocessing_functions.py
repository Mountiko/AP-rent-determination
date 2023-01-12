# third-party imports
import pandas as pd
import numpy as np

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
