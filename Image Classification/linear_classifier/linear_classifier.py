import numpy as np

class LinearClassifier(object):
    """
    Class for calculating the loss during training and
    Updating the weights
    """

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200, verbose=False):
        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, int(num_classes))
        # Run SGD to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            batch_indices = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W -= learning_rate * grad

            if verbose and it % 100:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))
        return loss_history
    
    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        X  is a numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.
        Returns predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        y_pred = np.zeros(X.shape[0])
        scores = X.dot(self.W)
        y_pred = scores.argmax(axis=1)
        return y_pred


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorizer(self.W, X_batch, y_batch, reg)
    


def svm_loss_vectorizer(W, X, y, reg):
    """
    W matrix that consists of the weights,
    input matrix X, target matrix y and reg the regularization strength. 
    The scores metric is calculated according to W.X.
    The loss is calculated from the average difference 
    between the true target matrix y and the predicted scores. 
    A further L2 regularization loss is added to encourage the weights to stay low.

    """

    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]

    scores = X.dot(W)
    y = [int(x) for x in y]
    correct_class_scores = scores[np.arange(num_train), y].reshape(num_train, 1)

    margin = np.maximum(0, scores - correct_class_scores + 1)
    margin[np.arange(num_train), y] = 0 # do not consider correct class in loss
    
    loss = margin.sum() / num_train
    loss += reg * np.sum(W*W)

    margin[margin > 0] = 1
    valid_margin = margin.sum(axis=1)
    margin[np.arange(num_train), y] -= valid_margin

    dW = (X.T).dot(margin) / num_train
    dW = dW + reg * 2 * W

    return loss, dW