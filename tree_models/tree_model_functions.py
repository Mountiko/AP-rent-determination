def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    errors = [
        100 * (abs(predictions[i] - y_test[i]) / y_test[i])
        for i in range(min(len(predictions), len(y_test)))
    ]
    count_good_predictions = sum(1 for i in errors if i <= 10)
    good_predictions = round(np.mean(100 * (count_good_predictions / len(errors))), 2)
    return good_predictions
    print(
        "Percentage of predictions with less than 10 % deviation: ",
        good_predictions,
        "%.",
    )
