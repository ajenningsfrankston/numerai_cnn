import numpy as np

def check_consistency(model,valid_data,train_data,poly,sf):
    eras = valid_data.era.unique()
    count = 0
    count_consistent = 0
    for era in eras:
        count += 1
        current_valid_data = valid_data[valid_data.era == era]
        features = [f for f in list(train_data) if "feature" in f]
        X_valid = current_valid_data[features]
        X_valid = poly.fit_transform(X_valid)
        X_valid = X_valid[:,sf]
        Y_valid = current_valid_data["target_bernie"]
        loss = model.evaluate(X_valid, Y_valid, batch_size=128, verbose=0)[0]
        if (loss < -np.log(.5)):
            consistent = True
            count_consistent += 1
        else:
            consistent = False
        print("{}: loss - {} consistent: {}".format(era, loss, consistent))
    print("Consistency: {}".format(count_consistent / count))


