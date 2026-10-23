"""
Support Vector Machine from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - standardize_features
import numpy as np

def standardize_features(x):
    # TODO: rescale each column of x to have mean 0 and std 1 (leave zero-std columns alone).
    x = x.astype(np.float64)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0] = 1.0
    return (x-mean)/std

# Step 2 - initialize_parameters
import numpy as np

def initialize_parameters(n_features):
    """Return a dict with 'w' of shape (n_features,) and scalar 'b'."""
    # TODO: create starting weights and bias for a linear SVM
    parameters = {'w':np.zeros(n_features),'b':0.0}
    return parameters

# Step 3 - compute_scores
import numpy as np

def compute_scores(x, params):
    """Return raw linear scores x @ w + b, shape (n_samples,)."""
    # TODO: score each example as a linear function of the current weights and bias.
    return x @ params['w'] + params['b']

# Step 4 - predict_from_scores
import numpy as np

def predict_from_scores(scores):
    # TODO: convert a 1-D array of raw scores into +1 / -1 class predictions.
    scores = np.where(scores>=0,+1,-1)
    return scores

# Step 5 - hinge_loss_example
def hinge_loss_example(score, y):
    # TODO: return the hinge loss for a single example with raw score `score` and label y in {-1, +1}.
    return max(0,1-y*score)

# Step 6 - svm_objective
def svm_objective(x, y, params, reg_lambda):
    """Return mean hinge loss over the dataset plus reg_lambda * (w dot w)."""
    scores = compute_scores(x, params)
    hinge_loss = np.maximum(0, 1 - y * scores)
    
    # Take the mean of the hinge loss and add the L2 regularization term
    return np.mean(hinge_loss) + reg_lambda * np.dot(params['w'], params['w'])

# Step 7 - compute_gradients
import numpy as np

def compute_gradients(x, y, params, reg_lambda):
    """Return {'dw': ndarray shape (n_features,), 'db': float} = gradient of svm_objective."""
    # TODO: compute the gradient of the SVM objective wrt params['w'] and params['b'].
    scores = compute_scores(x, params)
    margins = 1-y*scores
    subgradient_mask = np.where(margins>0,1,0)
    dw = (-1/x.shape[0])*(x.T)@(subgradient_mask*y) + 2*reg_lambda*params['w']
    db = (-1/x.shape[0])*np.sum(subgradient_mask*y)

    return {'dw':dw,'db':db}

# Step 8 - apply_update
def apply_update(params, grads, learning_rate):
    # TODO: return a new params dict after one gradient-descent step on 'w' and 'b'.
    w = params['w'] - grads['dw']*learning_rate
    b = params['b'] - grads['db']*learning_rate
    return {'w':w,'b':b}

# Step 9 - train_svm
def train_svm(x, y, learning_rate, reg_lambda, n_epochs):
    # TODO: fit a linear SVM by repeatedly updating parameters over n_epochs passes.
    params = initialize_parameters(x.shape[1])
    for i in range(n_epochs):
        grads = compute_gradients(x,y,params,reg_lambda)
        params = apply_update(params,grads,learning_rate)

    return params

# Step 10 - predict_labels
import numpy as np

def predict_labels(x, params):
    # TODO: return an array of {-1, +1} labels, one per row of x, using params['w'] and params['b'].
    scores = compute_scores(x,params)
    return predict_from_scores(scores)

# Step 11 - accuracy_score
import numpy as np

def accuracy_score(y_pred, y_true):
    # TODO: return the fraction of positions where y_pred equals y_true.
    array = np.where(y_pred == y_true,1,0)
    return array.mean()

