from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        loss_count = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i] #gradient SVM function 
                loss_count+= 1
        dW[:,y[i]] -= loss_count * X[i]

    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #add regularization to the gradient 
    dW += 2 * reg * W

    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    scores = np.dot(X,W)

    num_train = X.shape[0]
    num_classes = W.shape[1]

    correct_class_score = scores[list(range(num_train)),y].reshape(num_train,-1)
    scores += 1 - correct_class_score
    scores[list(range(num_train)),y] = 0 #set correct classes loss to zero 

    
    loss = np.sum(np.fmax(scores,0))
    loss /= num_train
    loss += reg * np.sum(W * W)

    scores_not_clamped = np.zeros_like(scores)
    scores_not_clamped[scores > 0] = 1 #take values that are not 0 

    scores_not_clamped[np.arange(num_train),y] = -np.sum(scores_not_clamped,axis=1) #regularization 
    dW = X.T.dot(scores_not_clamped)
    dW /= num_train
    dW += 2 * reg * W



    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
