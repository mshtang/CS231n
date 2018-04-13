import numpy as np
from random import shuffle
# from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        sumj = np.sum(np.exp(scores))

        def prob(k): return np.exp(scores[k])/sumj
        loss += -np.log(prob(y[i]))

        for k in range(num_classes):
            pk = prob(k)
            dW[:, k] += (pk - (k == y[i]))*X[i]

    loss /= num_train
    loss += .5 * reg * np.sum(W * W)
    dW /= num_train
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    f = X.dot(W)
    f -= np.max(f, axis=1, keepdims=True)
    sum_f = np.sum(np.exp(f), axis=1, keepdims=True)
    p = np.exp(f)/sum_f
    loss = np.sum(-np.log(p[np.arange(num_train), y]))

    idx = np.zeros_like(p)
    idx[np.arange(num_train), y] = 1
    dW = X.T.dot(p-idx)

    loss /= num_train
    loss += .5*reg*np.sum(W*W)
    dW /= num_train
    dW += reg*W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
