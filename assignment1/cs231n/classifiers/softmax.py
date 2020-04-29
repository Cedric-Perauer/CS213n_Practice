from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

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

    N = X.shape[0]
    C = W.shape[1]
    for i in range(N):
      scores = (W.T).dot(X[i].T) #W*X
      scores -= np.max(scores)
      y_true_score = scores[y[i]] #get score for the true class
      loss += -y_true_score  #score for the correct class
      class_score = 0 
      dWl = np.zeros_like(W)

      for j in range(C):
        class_score += np.exp(scores[j])
        if y[i] == j : #if current class is equal to label 
          dWl[:,j] +=  X[i] * (np.exp(scores[j])/np.sum(np.exp(scores))-1)
          
        else : 
          dWl[:,j] +=  X[i] * np.exp(scores[j])/np.sum(np.exp(scores))

      dW += dWl
      loss += np.log(class_score)

    loss += reg * np.sum(W*W)
    loss /= N
    dW += reg *W
    dW /= N
    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N = X.shape[0]
    C = W.shape[1]

    scores = np.dot(X,W)
    
    scores -= np.max(scores)
    y_true_score = np.choose(y,scores.T)
    softmax = np.exp(scores)/(np.sum(np.exp(scores),axis=1)).reshape(N,1)

    #gradient wrt to softmax 
    grad = softmax
    grad[range(N),y] -= 1


    #Backprop 
    dW += np.dot(X.T,grad) #backprop
    dW /= N
    dW += reg*W #regularize

    #Cross Enropy Loss
    loss = np.sum(- y_true_score + np.log(np.sum(np.exp(scores),axis=1)))
    loss /= N 
    loss += reg * np.sum(W*W)

    return loss, dW
