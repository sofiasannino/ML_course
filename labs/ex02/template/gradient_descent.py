# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
def compute_loss(y, tx, w, MSE=True):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    #sample size
    N=len(y)

    
    #compute the error matrix
    e=y- tx @ w 
    
    
    
    if MSE :
    # compute the MSE loss or MAE loss
        Lw= (0.5*N)*(e @ e) #e.T @ e
    else :
        #Lw= (1/N) * (np.sum(np.sum ( np.abs(e.reshape(-1, 1)) , axis=1), axis=0))
        Lw=(1/N) * sum(abs(e))

    return Lw

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    #sample size
    N=len(y)

    #compute error
    e=y - tx @ w

    #compute gradient
    grad=(-1/N)* (tx.T @ e)

    return grad

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # computing gradient and loss
       
        grad=compute_gradient(y, tx, w)

        loss=compute_loss(y, tx, w)
        
        # update w by gradient
        w = w - gamma * grad

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return losses, ws
