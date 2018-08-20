#
# group correlated features and change the order
#
# cnn to deep classifier
#


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from check_consistency import check_consistency


print("# Loading data...")
# The training data is used to train your model how to predict the targets.
train = pd.read_csv('~/numerai_datasets/numerai_training_data.csv', header=0)
# The tournament data is the data that Numerai uses to evaluate your model.
tournament = pd.read_csv('~/numerai_datasets/numerai_tournament_data.csv', header=0)

# The tournament data contains validation data, test data and live data.
# Validation is used to test your model locally so we separate that.

validation = tournament[tournament['data_type'] == 'validation']

# There are five targets in the training data which you can choose to model using the features.
# Numerai does not say what the features mean but that's fine; we can still build a model.
# Here we select the bernie_target.
train_bernie = train.drop([
    'id', 'era', 'data_type',
    'target_charles', 'target_elizabeth',
    'target_jordan', 'target_ken'], axis=1)

# Transform the loaded CSV data into numpy arrays

features = [f for f in list(train_bernie) if "feature" in f]
X = train_bernie[features]
Y = train_bernie['target_bernie']
x_prediction = validation[features]
ids = tournament['id']

# explore stacking

print("# ridge regression ")

from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score

clf = RidgeClassifier(alpha=1.0)
clf.fit(X.values, Y.values)

#check_consistency(clf, validation, train)


# Naive Bayes

print("# Naive Bayes")

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X.values, Y.values)

#check_consistency(gnb, validation, train)

# set parameters:

batch_size = 256
epochs = 8


def create_model(neurons=50, dropout=0.2):
    model = Sequential()
    # we add a vanilla hidden layer:
    model.add(Dense(neurons))
    model.add(Activation('sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(neurons))
    model.add(Activation('sigmoid'))
    model.add(Dropout(dropout))

    model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy', 'accuracy'])
    return model


model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)

neurons = [32, 40]
dropout = [0.1, 0.2]
param_grid = dict(neurons=neurons, dropout=dropout)

gkf = GroupKFold(n_splits=5)
kfold_split = gkf.split(X, Y, groups=train.era)

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold_split, scoring='neg_log_loss', n_jobs=4, verbose=3)
grid_result = grid.fit(X.values, Y.values)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# check consistency

check_consistency(grid.best_estimator_.model, validation, train)

# create predictions


x_prediction = tournament[features]
t_id = tournament["id"]
y_prediction = grid.best_estimator_.model.predict_proba(x_prediction.values, batch_size=batch_size)
results = np.reshape(y_prediction, -1)
results_df = pd.DataFrame(data={'probability_bernie': results})
joined = pd.DataFrame(t_id).join(results_df)

filename = 'predictions.csv'
path = '~/numerai_predictions/' + filename
print()
print("Writing predictions to " + path.strip())
# # Save the predictions out to a CSV file
joined.to_csv(path, float_format='%.5f', index=False)
