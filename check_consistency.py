import numpy as np
from sklearn.metrics import accuracy_score


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
        prediction = model.predict_proba(x_valid.values)
        prediction[prediction >= 0.5] = 1
        prediction[prediction < 0.5] = 0
        accuracy = accuracy_score(y_valid.values,prediction)
        if accuracy > 0.5 :
            consistent = True
            count_consistent += 1
        else:
            consistent = False
        print("{}: accuracy - {} consistent: {}".format(era, accuracy, consistent))
    print("Consistency: {}".format(count_consistent / count))


