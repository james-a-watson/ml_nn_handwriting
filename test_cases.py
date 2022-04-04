import logging
import random

import numpy as np
from scipy import io

import ml_functions as mlf
import debug as dbg


logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger('ML-NN')

##############################################
###  Load Training Data and Set Constants  ###
##############################################
input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10

data = io.loadmat('learning_data/data.mat')
X = data['X']
y = data['y']

def check_cost_function():
    logger.info('Feedforward Using Neural Network ...')
    Lambda = 0
    nn_params = mlf.unroll_params(io.loadmat('learning_data/weights.mat'))
    J, grad = mlf.nnCostFunction(nn_params, 
                        input_layer_size, 
                        hidden_layer_size, 
                        num_labels, 
                        X, 
                        y, 
                        Lambda)
    logger.info(f'Cost at parameters (exp~ 0.287629): {J}')

    logger.info('Checking Cost Function (w/ Regularization) ...')
    Lambda = 1
    nn_params = mlf.unroll_params(io.loadmat('learning_data/weights.mat'))
    J, grad = mlf.nnCostFunction(nn_params, 
                        input_layer_size, 
                        hidden_layer_size,
                        num_labels, 
                        X, 
                        y, 
                        Lambda)
    logger.info(f'Cost at parameters (exp~  0.383770): {J}')


def check_backprop():
    logger.info('Checking Backpropagation...')
    diff = dbg.checkNNGradients()
    logger.info(f"Relative Difference: {diff} (should be less than 1e-9)")


def check_regularization():
    ####################################
    ###    Implement Regularization  ###
    ####################################
    logger.info(f"Checking Backpropagation (w/ Regularization)...")
    Lambda = 3 
    diff = dbg.checkNNGradients(Lambda)
    logger.info(f"Relative Difference: {diff} (should be less than 1e-9)")

    nn_params = mlf.unroll_params(io.loadmat('learning_data/weights.mat'))
    debug_J, debug_grad  = mlf.nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
    logger.info(f'Cost at debugging params (lambda={Lambda}) (exp~ 0.576051): {debug_J} ')


def full_test():
    check_cost_function()
    check_backprop()
    check_regularization()