import warnings
warnings.filterwarnings("ignore")

import tensorflow.keras as keras
keras.__version__

from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_scaled = StandardScaler().fit_transform(cancer.data)
print("Original data (rows, features):", X_scaled.shape)

# TIME
# Generating to polynomial features is not that time consuming
poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(X_scaled)
print("All polynomial features (order 2):", X_poly.shape)

# TIME
# A fairly generic random forest
rfc = RandomForestClassifier(max_depth=7, n_estimators=10, random_state=1)

# Do some work to pick the optimal number of features
# "Recursive feature elimination using cross-validation"
rfecv = RFECV(estimator=rfc, cv=5, n_jobs=-1)
X_poly_top = rfecv.fit_transform(X_poly, cancer.target)

# The "top" features selected for the model
print("Best polynomial features", X_poly_top.shape)

# TIME
# Do a train/test split on the "poly_top" features
X_train, X_test, y_train, y_test = train_test_split(
    X_poly_top, cancer.target, random_state=42)

# Train the selected RFC model
rfc = RandomForestClassifier(max_depth=7, n_estimators=10, random_state=1)
print("Test accuracy:", rfc.fit(X_train, y_train).score(X_test, y_test))

print("---- START Pre-amble ----")

batch_size = 32
in_dim = cancer.data.shape[1]
hidden1 = X_poly_top.shape[1]   # The size of layer that deduces poly features
hidden2 = 20                    # The size of the \"inference layer\"
out_dim = 1                     # Output a single value

batches_in_data = X_train.shape[0]/batch_size
epochs = int(5000/batches_in_data)
learning_rate = 1e-4

# Split the original data
X_train, X_test, y_train, y_test = train_test_split(
                           cancer.data, cancer.target, random_state=42)
cancer.data.shape   # The shape of the data being split"

print("---- START tensorflow Keras ----")

model_k = keras.models.Sequential([
    # This layer allows "polynomial features"
    keras.layers.Dense(hidden1, activation='relu', input_shape=(in_dim,)),
    # This layer is the essential "inference"
    keras.layers.Dense(hidden2),
    # Often Leaky ReLU eliminates the "dead neuron" danger
    keras.layers.LeakyReLU(),
    # A Dropout layer sometimes reduces co-adaptation of neurons
    keras.layers.Dropout(rate=0.25),
    # A sigmoid activation is used for a binary decision
    keras.layers.Dense(out_dim, activation='sigmoid')
])
model_k.summary()

print("---- START tensorflow Keras ADAM ----")

# TIME BLOCK
# Sometimes we do better using Adaptive Moment Optimization
model_k.compile(loss='mean_squared_error',
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                metrics=['accuracy'])
history = model_k.fit(X_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=False,
                      validation_data=(X_test, y_test))
score = model_k.evaluate(X_test, y_test, verbose=True)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
