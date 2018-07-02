#
# group correlated features and change the order
#
# cnn to deep classifier
#


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from check_consistency import check_consistency
from re_order_features import sort_features

training_data = pd.read_csv('~/numerai_datasets/numerai_training_data.csv', header=0)
tournament_data = pd.read_csv('~/numerai_datasets/numerai_tournament_data.csv', header=0)

validation_data = tournament_data[tournament_data.data_type=='validation']
complete_training_data = training_data

features = [f for f in list(complete_training_data) if "feature" in f]
X = complete_training_data[features]
Y = complete_training_data["target"]


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold


from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Conv1D, GlobalMaxPooling1D

# set parameters:

batch_size = 32
filters = 50
kernel_size = 3
epochs = 8


def create_model(neurons=50, dropout=0.2):
    model = Sequential()
    # we add a vanilla hidden layer:
    model.add(Dense(neurons))
    model.add(Activation('sigmoid'))
    model.add(Dropout(0.2))

    model.add(Dense(neurons))
    model.add(Activation('sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='glorot_normal'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_crossentropy', 'accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)


neurons = [16, 32]
dropout = [0.1, 0.3]
param_grid = dict(neurons=neurons, dropout=dropout)

gkf = GroupKFold(n_splits=5)
kfold_split = gkf.split(X, Y, groups=complete_training_data.era)

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=kfold_split, scoring='neg_log_loss',n_jobs=1, verbose=3)
grid_result = grid.fit(X.values, Y.values)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# check consistency

check_consistency(grid.best_estimator_.model, validation_data,complete_training_data)

# create predictions
from time import strftime,gmtime

x_prediction = tournament_data[features]
t_id = tournament_data["id"]
y_prediction = grid.best_estimator_.model.predict_proba(x_prediction.values, batch_size=128)
results = np.reshape(y_prediction,-1)
results_df = pd.DataFrame(data={'probability':results})
joined = pd.DataFrame(t_id).join(results_df)
# path = "predictions_w_loss_0_" + '{:4.0f}'.format(history.history['loss'][-1]*10000) + ".csv"
filename = 'predictions_{:}'.format(strftime("%Y-%m-%d_%Hh%Mm%Ss", gmtime())) + '.csv'
path = '~/numerai_predictions/' +filename
print()
print("Writing predictions to " + path.strip())
# # Save the predictions out to a CSV file
joined.to_csv(path,float_format='%.15f', index=False)
