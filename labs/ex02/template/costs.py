# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w, MSE= True):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """

    #sample size
    N=len(y)
    
    #compute the error matrix
    e=y- tx @ w 

    # compute the MSE loss or MAE loss
    if MSE :
        Lw= (0.5*N)*(e @ e) #e.T @ e
    else :
        #Lw= (1/N) * (np.sum(np.sum ( np.abs(e.reshape(-1, 1)) , axis=1), axis=0))
        Lw=(1/N) * sum(abs(e))

    return Lw
    
