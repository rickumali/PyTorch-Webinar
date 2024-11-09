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

# # Train the selected RFC model
# rfc = RandomForestClassifier(max_depth=7, n_estimators=10, random_state=1)
# print("Test accuracy:", rfc.fit(X_train, y_train).score(X_test, y_test))

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

print("---- START PyTorch ----")
import numpy as np
import torch
from torch.autograd import Variable

# Create a sequential Neural Network
model_t = torch.nn.Sequential(
    # This layer allows "polynomial features"
    torch.nn.Linear(in_dim, hidden1),
    # The activation is treated as a separate layer
    torch.nn.ReLU(),
    # This layer is the essential "inference"
    torch.nn.Linear(hidden1, hidden2),
    # Often Leaky ReLU eliminates the "dead neuron" danger
    torch.nn.LeakyReLU(), 
    # A Dropout layer sometimes reduces co-adaptation of neurons
    torch.nn.Dropout(p=0.25),
    # A sigmoid activation is used for a binary decision
    torch.nn.Linear(hidden2, out_dim),  
    torch.nn.Sigmoid()
)
print(model_t)

from torch import device, cuda
from torchsummary import summary

# torchsummary has a glitch. If running on a CUDA-enabled build
# it only wants to print a CUDA model
if cuda.is_available():
    print("CUDA is available!")
    model_t = model_t.to(device('cuda'))
else:
    print("CUDA is NOT available!")
    # Do the to(device('cpu')) stuff?
    # model_t = model_t.to(device('cpu'))
    
summary(model_t, input_size=(1,in_dim))

print("---- SETUP Training ----")
show_every = 250

def do_training():
    for t in range(5000):
        # Forward pass: compute predicted y by passing x to the model.
        y_pred = model_t(X)

        # Compute and print loss.
        loss = loss_fn(y_pred, y)
        if not t % show_every:
            y_test_pred = model_t(Variable(X_test_T))
            prediction = [int(x > 0.5) 
                          for x in y_test_pred.data.cpu().numpy()]
            test_accuracy = (prediction == y_test).sum() / len(y_test)
            train_pred = [int(x > 0.5) 
                          for x in y_pred.data.cpu().numpy()]
            train_accuracy = (train_pred == y_train).sum() / len(y_train)
            print("Batch: %04d | Training Loss: %6.2f | Train accuracy: %.4f | Test accuracy: %.4f" % (
                          t, loss.item(), train_accuracy, test_accuracy))

        # Before the backward pass, use the optimizer object to zero all of the
        # gradients for the variables it will update (which are the learnable
        # weights of the model). This is because by default, gradients are
        # accumulated in buffers( i.e, not overwritten) whenever .backward()
        # is called. Checkout docs of torch.autograd.backward for more details.
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss with respect to model
        # parameters
        loss.backward()

        # Calling the step function on an Optimizer makes an update to its
        # parameters
        optimizer.step()

print("---- START Training/Model (ADAM) ----")
## Now run model
X = torch.from_numpy(X_train).float()
y = torch.from_numpy(y_train[:, np.newaxis]).float()
X_test_T = torch.from_numpy(X_test).float()
y_test_T = torch.from_numpy(y_test[:, np.newaxis]).float()

if cuda.is_available():
    X = X.to(device('cuda'))
    y = y.to(device('cuda'))
    X_test_T = X_test_T.to(device('cuda'))
    y_test_T = y_test_T.to(device('cuda'))

loss_fn = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model_t.parameters(), lr=learning_rate)
do_training()

print("---- PREDICTIONS ADAM ----")
predictions = model_t(X_test_T[:10])
for row, prediction in enumerate(predictions):
    print("Observation %d; probability benign: %0.3f%%" % (row, prediction*100))

# TIME BLOCK
