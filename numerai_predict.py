#
# group correlated features and change the order
#
# cnn to deep classifier
#


import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from check_consistency import check_consistency

from pridge_classifier import PRidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier


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


rdc = PRidgeClassifier(alpha=1.0)
lrc = LogisticRegression()

rdc.fit(X.values,Y.values)
lrc.fit(X.values,Y.values)




# keras parameters

batch_size = 256
epochs = 16


def create_model(neurons=20, dropout=0.1):
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


keras_model = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=2)

keras_model.fit(X.values,Y.values)


model_list = [lrc,rdc,keras_model]

consistencies = [ check_consistency(model, validation, train) for model in model_list]

print("Consistencies: ",format(consistencies))

most_consistent_model = model_list[consistencies.index(max(consistencies))]


# bagging classifier


model = BaggingClassifier(base_estimator=most_consistent_model, n_estimators=10)
model.fit(X.values,Y.values)


# check consistency
final_model = model
consistency = check_consistency(final_model, validation, train)
print("Consistency: {}".format(consistency))


x_prediction = tournament[features]
t_id = tournament["id"]
raw_predict = final_model.predict_proba(x_prediction.values)
y_prediction = raw_predict[:,1]
results = np.reshape(y_prediction, -1)
results_df = pd.DataFrame(data={'probability_bernie': results})
joined = pd.DataFrame(t_id).join(results_df)

filename = 'predictions.csv'
path = '~/numerai_predictions/' + filename
print()
print("Writing predictions to " + path.strip())
# # Save the predictions out to a CSV file
joined.to_csv(path, float_format='%.5f', index=False)
