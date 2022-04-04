import logging
import math
from scipy import io

import numpy as np
from matplotlib import pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

for name in ['matplotlib.font_manager', 'PIL.PngImagePlugin']:
    logging.getLogger(name).disabled = True

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger('ML-NN')

def unroll_params(nn_params):
    Theta1 = nn_params['Theta1']
    Theta2 = nn_params['Theta2']

    # Unroll parameters
    it1_ravel = np.array([Theta1.ravel()])
    it2_ravel = np.array([Theta2.ravel()])
    nn_params = np.concatenate((np.transpose(it1_ravel), np.transpose(it2_ravel)))

    return nn_params


def display_data(X, example_width=False):
    ###############################
    ###  Building Display Area  ###
    ###############################
    X = np.array(X)
    if not example_width:
        example_width = round(math.sqrt(len(X[0])))

    m = len(X) # Number of rows in sample set X
    n = len(X[0]) # Number of columns in sample set X

    example_height = int(n/example_width)

    # Dimensions of the final display. Number of panels
    display_rows = math.floor(math.sqrt(m))
    display_cols = math.ceil(m / display_rows)

    pad = 1 # Padding between panels

    # Total display pixel - Dimensions include padding between panels and around border
    # e.g 10x10 panels, 20x20 pixels, 11x11 padding = 211x211 pixels
    display_pixel_cols = pad + display_rows * (example_height + pad)
    display_pixel_rows = pad + display_cols * (example_width + pad)
    display_array = -1 * np.ones((int(display_pixel_rows), int(display_pixel_cols)))


    ###############################
    ###     Creating Panels     ###
    ###############################
    curr_ex = 0
    for j in range(1, display_rows + 1):
        for i in range(1, display_cols + 1):
            # Reshaping raw data into display panel
            X_abs = np.absolute(X[curr_ex,:])
            max_val = X_abs.max()
            display_chunk = np.reshape(X[curr_ex, :], (example_height, example_width))/max_val

            # Inserting panel into display area
            v1 = int(pad + (j - 1) * (example_height + pad))
            v2 = int(example_height + pad + (j - 1) * (example_height + pad))
            w1 = int(pad + (i - 1) * (example_width + pad))
            w2 = int(example_width + pad + (i - 1) * (example_width + pad))
            display_array[v1:v2, w1:w2] = display_chunk
            curr_ex += 1

    # Plotting display
    plt.imshow(np.transpose(display_array))
    plt.show()


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def sigmoidGradient(x):
    return sigmoid(x)*(1 - sigmoid(x))


def randInitializeWeights(L_in, L_out):
    epsilon_init = 0.12
    
    W = np.random.random_sample((L_out, 1 + L_in)) * 2 * epsilon_init - epsilon_init
    
    return W


def initiallise_params(input_layer_size, hidden_layer_size, num_labels):
    logger.info('Initializing Neural Network Parameters ...')
    initial_nn_params = {}
    initial_nn_params["Theta1"] = randInitializeWeights(input_layer_size, hidden_layer_size)
    initial_nn_params["Theta2"] = randInitializeWeights(hidden_layer_size, num_labels)

    initial_params = unroll_params(initial_nn_params)

    return initial_params


def unravel_nn_params(
    nn_params, 
    input_layer_size=400, 
    hidden_layer_size=25, 
    num_labels=10):

    split_theta = ((hidden_layer_size) * (input_layer_size + 1))

    Theta1 = np.reshape(nn_params[0:split_theta], (hidden_layer_size, input_layer_size + 1))
    Theta2 = np.reshape(nn_params[split_theta:], (num_labels, hidden_layer_size + 1))

    return Theta1, Theta2


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    m, n = np.shape(X)

    Theta1, Theta2 = unravel_nn_params(nn_params, input_layer_size, hidden_layer_size, num_labels,)

    Theta1_grad = np.zeros(np.shape(Theta1))
    Theta2_grad = np.zeros(np.shape(Theta2))
    
    # Layer 1 -> Layer 2
    a1 = np.append(np.ones((m, 1)), X, axis=1)
    z2 = np.matmul(a1, np.transpose(Theta1))
    a2 = sigmoid(z2)
    
    # # Layer 1 -> Layer 2
    a2 = np.append(np.ones((m, 1)), a2, axis=1)
    z3 = np.matmul(a2, np.transpose(Theta2))
    h = sigmoid(z3)

    # Recode y vector
    y_uncoded = y - 1
    y = np.zeros((m, num_labels))
    x = 0
    for a in y_uncoded:
        y[x][a] = 1 
        x += 1

    # Compute unregularized Cost J
    p = (-y)*np.log(h)
    q = (1 - y)*np.log(1 - h)
    J = (1/m)*np.sum(np.sum(p - q))

    # Compute regularized cost
    Theta1_reg = Theta1
    Theta1_reg[:, 0] =  0
    Theta2_reg = Theta2
    Theta2_reg[:, 0] =  0

    I = np.sum(np.sum(np.multiply(Theta1_reg, Theta1))) + np.sum(np.sum(np.multiply(Theta2_reg, Theta2)))
    J = J + (Lambda*I)/(2*m)

    # Back Prop Vecotrized
    at1 = np.append(np.ones((m, 1)), X, axis=1)
    zt2 = np.matmul(at1, np.transpose(Theta1))
    at2 = sigmoid(zt2)
    at2 = np.append(np.ones((m, 1)), at2, axis=1)
    zt3 = np.matmul(at2, np.transpose(Theta2))
    at3 = sigmoid(zt3)

    d3 = (at3 - y)
        
    zt2 = np.append(np.ones((m, 1)), zt2, axis=1)
    d2 = np.matmul(d3, Theta2)*sigmoidGradient(zt2)
    d2 = d2[:,1:]
        
    Theta2_grad = Theta2_grad + np.matmul(np.transpose(d3), at2)
    Theta1_grad = Theta1_grad + np.matmul(np.transpose(d2), at1)

    Theta2[:, 0] = 0
    Theta1[:, 0] = 0

    Theta2_grad = Theta2_grad/m + (Lambda/m)*Theta2
    Theta1_grad = Theta1_grad/m + (Lambda/m)*Theta1

    # Unroll parameters
    t1g_ravel = np.array([Theta1_grad.ravel()])
    t2g_ravel = np.array([Theta2_grad.ravel()])
    grad = np.concatenate((np.transpose(t1g_ravel), np.transpose(t2g_ravel)))

    return J, grad


def fmincg(X, params, length):
    ###############################
    ###  Line Search Constants  ###  
    ###############################
    # RHO and SIG are the constants in the Wolfe-Powell conditions
    RHO = 0.01 
    SIG = 0.5  
    INT = 0.1       # Don't reevaluate within 0.1 of the limit of the current bracket
    EXT = 3.0       # Extrapolate maximum 3 times the current bracket
    MAX = 20        # Max 20 function evaluations per line search
    RATIO = 100     # Maximum allowed slope ratio

    cost_list = []                    
    cost1, grad1 = nnCostFunction(X, **params)   # get function value and gradient
    s = -grad1                                   # search direction is steepest
    d1 = np.matmul(np.transpose(-s), s)          # this is the slope
    z1 = 1/(1-d1)                                # initial step is 1/(|s|+1)

    # Initilalise the loop counter and line search bool
    i = 0
    line_search_failed = False
    while i < length:
        i += 1
        copy_X, copy_cost, copy_grad = X, cost1, grad1 # Copy current values

        ###############################
        ###    Start Line Search    ###  
        ###############################
        X = X + z1*s    
        cost2, grad2 = nnCostFunction(X, **params)
        
        d2 = np.matmul(np.transpose(grad2), s)
        f3 = cost1 
        d3 = d1 
        z3 = -z1             # initialize point 3 equal to point 1

        if length > 0: 
            M = MAX
        else: 
            M = np.minimum(MAX, -length-i)

        success = False
        limit = -1 
        while True:
            while ((cost2 > cost1+z1*RHO*d1) or (d2 > -SIG*d1)) and (M > 0):
                limit = z1      # tighten the bracket
                if cost2 > cost1:
                    z2 = z3 - (0.5*d3*z3*z3)/(d3*z3+cost2-f3)                
                else:
                    A = 6*(cost2-f3)/z3+3*(d2+d3)  # cubic fit
                    B = 3*(f3-cost2)-z3*(d3+2*d2)
                    try:
                        z2 = (math.sqrt(B*B-A*d2*z3*z3)-B)/A  # numerical error possible - ok!
                    except ValueError:
                        z2 = z3/2                             # if we had a numerical problem then bisect
                    
                z2 = np.maximum(np.minimum(z2, INT*z3),(1-INT)*z3)  # don't accept too close to limits
                z1 = z1 + z2                                # update the step
                X = X + z2*s
                cost2, grad2 = nnCostFunction(X, **params)
                M = M - 1; i = i + (length<0)
                d2 = np.matmul(np.transpose(grad2), s)
                z3 = z3-z2          #  z3 is now relative to the location of z2


            if cost2 > cost1+z1*RHO*d1 or d2 > -SIG*d1:
                break                               # this is a failure
            elif d2 > SIG*d1:
                success = True
                break              
            elif M == 0:
                break
            
            A = 6*(cost2-f3)/z3+3*(d2+d3)              # make cubic extrapolation
            B = 3*(f3-cost2)-z3*(d3+2*d2)
            try:
                z2 = -d2*z3*z3/(B+math.sqrt(B*B-A*d2*z3*z3)) # num. error possible - ok!
            except ValueError:
                if limit < -0.5:             # if we have no upper limit
                    z2 = z1 * (EXT-1)       # the extrapolate the maximum amount
                else:
                    z2 = (limit-z1)/2       # otherwise bisect
                    
            if (limit > -0.5) and (z2+z1 > limit):    # extraplation beyond max?
                z2 = (limit-z1)/2;                      # bisect
            elif (limit < -0.5) and (z2+z1 > z1*EXT):   # extrapolation beyond limit
                z2 = z1*(EXT-1.0)                         # set to extrapolation limit
            elif z2 < -z3*INT:
                z2 = -z3*INT
            elif (limit > -0.5) and (z2 < (limit-z1)*(1.0-INT)): # too close to limit?
                z2 = (limit-z1)*(1.0-INT)

            f3 = cost2; d3 = d2; z3 = -z2      # set point 3 equal to point 2
            z1 = z1 + z2; X = X + z2*s;     # update current estimates
            cost2, grad2 = nnCostFunction(X, **params)
            M = M - 1; i = i + (length<0)
            d2 = np.matmul(np.transpose(grad2), s)


        if success:                              # if line search succeeded
            cost1 = cost2
            np.append(cost_list, [cost1])

            cost_list = np.transpose(cost_list)
            print(f'Iteration {i} | Cost: {cost1}')
            s_num = (np.matmul(np.transpose(grad2),grad2)-np.matmul(np.transpose(grad1),grad2))
            s_dnom = (np.matmul(np.transpose(grad1),grad1))*s - grad2      
            s = s_num/s_dnom            # Polack-Ribiere direction
            
            tmp = grad1
            grad1 = grad2
            grad2 = tmp   # swap derivatives

            d2 = np.matmul(np.transpose(grad1),s)
            if d2 > 0:          # new slope must be negative
                s = -grad1        # otherwise use steepest direction
                d2 = -np.matmul(np.transpose(s),s) 

            z1 = z1 * np.minimum(RATIO, d1/(d2-np.finfo(float).tiny))   # slope ratio but max RATIO
            d1 = d2
            line_search_failed = False      # this line search did not fail
        else: # Line search failed
            # restore point from before failed line search
            X, cost1, grad1 = copy_X, copy_cost, copy_grad

            if line_search_failed or i > np.absolute(length):    # line search failed twice in a row
                break                                  # or we ran out of time, so we give up

            tmp = grad1 
            grad1 = grad2 
            grad2 = tmp       # swap derivatives
            s = -grad1        # try steepest
            d1 = -np.matmul(np.transpose(s),s)
            z1 = 1/(1-d1)                     
            line_search_failed = True      # this line search failed

    return X, cost_list, i


def predict(Theta1, Theta2, X):
    m, n = np.shape(X)
    num_labels, l = np.shape(Theta2)

    p = np.zeros((m, 1))

    q1 = np.append(np.ones((m, 1)), X, axis=1)
    h1 = sigmoid(np.matmul(q1, np.transpose(Theta1)))

    q2 = np.append(np.ones((m, 1)), h1, axis=1)
    h2 = sigmoid(np.matmul(q2, np.transpose(Theta2)))

    # Get the digit that scored the max (predicted digit) from h2 
    p = np.argmax(h2,axis=1)
    p = p + 1
    p = p[np.newaxis]
    p = np.transpose(p)

    return p, m