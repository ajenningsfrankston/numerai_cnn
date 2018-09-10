
from sklearn.metrics import log_loss


BENCHMARK = 0.693


def check_consistency(model,valid_data,train_data):
    eras = valid_data.era.unique()
    count = 0
    count_consistent = 0
    for era in eras:
        count += 1
        current_valid_data = valid_data[valid_data.era == era]
        features = [f for f in list(train_data) if "feature" in f]
        x_valid = current_valid_data[features]
        y_valid = current_valid_data["target_bernie"]
        raw_prediction = model.predict_proba(x_valid.values)
        logloss = log_loss(y_valid.values,raw_prediction)
        if logloss < BENCHMARK:
            count_consistent += 1
        consistency = (count_consistent/count)*100
    return consistency


