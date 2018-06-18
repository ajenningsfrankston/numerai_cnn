import numpy as np

def check_consistency(model,valid_data,train_data):
    eras = valid_data.era.unique()
    count = 0
    count_consistent = 0
    for era in eras:
        count += 1
        current_valid_data = valid_data[valid_data.era == era]
        features = [f for f in list(train_data) if "feature" in f]
        X_valid = current_valid_data[features]
        Y_valid = current_valid_data["target"]
        loss = model.evaluate(X_valid.values, Y_valid.values, batch_size=128, verbose=0)[0]
        if (loss < -np.log(.5)):
            consistent = True
            count_consistent += 1
        else:
            consistent = False
        print("{}: loss - {} consistent: {}".format(era, loss, consistent))
    print("Consistency: {}".format(count_consistent / count))
