import logging
import math

import numpy as np
from numpy import linalg
from scipy import io

import ml_functions as mlf

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger('ML-NN')

input_layer_size = 3
hidden_layer_size = 5
num_labels = 3
m_ = 5

def debugInitializeWeights(fan_out, fan_in):
    W = np.zeros((fan_out, 1 + fan_in))
    
    n, m = np.shape(W)
    S = np.arange(1, n*m + 1)

    # Initialize W with "sin" ensures W is always of the same values (good for debugging)
    W = np.reshape(np.sin(S), (n, m)) / 10

    return W


def computeNumericalGradient(nn_params_db, X, y, Lambda):
    Theta = nn_params_db

    n, m = np.shape(Theta)

    numgrad = np.zeros((n,m))
    perturb = np.zeros((n,m))

    e = 0.0001

    for p in range(0, n*m):
        perturb[p] = e
        pTheta = Theta - perturb
        loss1, grad1 = mlf.nnCostFunction(pTheta, 
                                input_layer_size, 
                                hidden_layer_size, 
                                num_labels, 
                                X, 
                                y, 
                                Lambda)

        pTheta = Theta + perturb
        loss2, grad2 = mlf.nnCostFunction(pTheta, 
                                input_layer_size, 
                                hidden_layer_size, 
                                num_labels, 
                                X, 
                                y, 
                                Lambda)
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0

    return numgrad


def checkNNGradients(Lambda=0):
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m_, input_layer_size - 1)

    y = 1 + np.mod(np.arange(1, m_ + 1), num_labels)
    y = np.transpose(np.array([y]))

    # Unroll parameters
    it1_ravel = np.array([Theta1.ravel()])
    it2_ravel = np.array([Theta2.ravel()])
    nn_params_db = np.concatenate((np.transpose(it1_ravel), np.transpose(it2_ravel)))

    cost, grad = mlf.nnCostFunction(nn_params_db, 
                                    input_layer_size,
                                    hidden_layer_size,
                                    num_labels,
                                    X,
                                    y,
                                    Lambda)

    numgrad = computeNumericalGradient(nn_params_db, X, y, Lambda)
    
    n, m = np.shape(numgrad)

    diff = linalg.norm(numgrad-grad)/linalg.norm(numgrad+grad)

    return diff
