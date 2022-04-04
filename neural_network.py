import logging
import random

import numpy as np
from scipy import io

import ml_functions as mlf
import test_cases

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger('ML-NN')

logger.info('Loading Data ...')
data = io.loadmat('learning_data/data.mat')
X = data['X']
y = data['y']


def profile():
    import cProfile
    import io
    import math
    import pstats

    # nn_params = mlf.initiallise_params(400, 25, 10)
    
    with cProfile.Profile() as pr:
        # mlf.nnCostFunction(nn_params, 400, 25, 10, X, y, 1)
        train_neural_network(150)
    
    result = io.StringIO()
    pstats.Stats(pr, stream=result).print_stats()
    result = result.getvalue()
    result='ncalls'+result.split('ncalls')[-1]
    result='\n'.join([','.join(line.rstrip().split(None,5)) for line in result.split('\n')])
    
    with open('test.csv', 'w+') as f:
        f.write(result)
        f.close()


def visualise_data():
    logger.info('Visualizing Data ...')

    sel = random.sample(range(len(X)), 100)
    X_sel = [X[index] for index in sel]

    mlf.display_data(X_sel)


def train_neural_network(iterations=100, input_layer_size=400, hidden_layer_size=25, num_labels=10):
    initial_params = mlf.initiallise_params(input_layer_size, hidden_layer_size, num_labels)

    logger.info(f"Training Neural Network - {iterations} Iterations...")
    fmincg_params = {
        "input_layer_size": input_layer_size,
        "hidden_layer_size": hidden_layer_size,
        "num_labels": num_labels,
        "X": X,
        "y": y,
        "Lambda": 1
    }

    nn_params, cost, i = mlf.fmincg(initial_params, fmincg_params, iterations)

    logger.info(f"Visualizing Neural Network...")
    Theta1, Theta2 = mlf.unravel_nn_params(nn_params)
    mlf.display_data(Theta1[:, 1:])

    logger.info(f"Predicting Labelled Data...")
    preds, number_of_preds = mlf.predict(Theta1, Theta2, X)
    percent_correct = (np.sum(preds == y)/number_of_preds)*100
    logger.info(f"Training Accuracy: {percent_correct}%")


visualise_data()
test_cases.full_test()
train_neural_network(150)
# profile()
