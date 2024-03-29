import numpy as np
import pandas as pd


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    errors = [
        100 * (abs(predictions[i] - y_test[i]) / y_test[i])
        for i in range(min(len(predictions), len(y_test)))
    ]
    count_good_predictions = sum(1 for i in errors if i <= 10)
    good_predictions = round(np.mean(100 * (count_good_predictions / len(errors))), 2)
    return good_predictions


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


# Evaluate model score
def evaluate_prediction_score(state_prediction_score):
    state_prediction_df = pd.DataFrame(
        state_prediction_score, columns=["state", "score", "inserates"]
    )
    number_of_inserates = state_prediction_df["inserates"].sum()
    state_prediction_df["weighted_score"] = (
        state_prediction_df["score"] * state_prediction_df["inserates"]
    )
    prediction_score_weighted = (
        state_prediction_df["weighted_score"].sum() / number_of_inserates
    )

    return prediction_score_weighted
