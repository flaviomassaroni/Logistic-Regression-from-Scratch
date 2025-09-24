import numpy as np

class LogisticRegression:
    
    """
    Logistic Regression model.

    Parameters:
        learning_rate (float): Learning rate for the model.

    Methods:
        initialize_parameter(): Initializes the parameters of the model.
        sigmoid(z): Computes the sigmoid activation function for given input z.
        forward(X): Computes forward propagation for given input X.
        compute_cost(predictions): Computes the cost function for given predictions.
        compute_gradient(predictions): Computes the gradients for the model using given predictions.
        fit(X, y, iterations, plot_cost): Trains the model on given input X and labels y for specified iterations.
        predict(X): Predicts the labels for given input X.
    """

    def __init__(self, learning_rate):
        np.random.seed(1)
        self.learning_rate = learning_rate

    def initialize_parameters(self):
        self.W = np.zeros(self.X.shape[1])
        self.b = 0.0

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward(self, X):
        Z = np.dot(X, self.W) + self.b
        A = self.sigmoid(Z)
        return A

    def compute_cost(self, predictions):
        
        m = self.X.shape[0]

        # Computing Binary cross entropy cost function
        cost = np.sum((-np.log(predictions + 1e-8)*self.y) + (-np.log(1 - predictions + 1e-8))* (1 - self.y))
        cost = cost / m
        return cost

    def compute_gradients(self, predictions):
        pass

    def fit(self, X, y, iterations, plot_cost = True):
        """
        Trains the model on given input X and labels y for specified iterations.

        Parameters:
            X (numpy.ndarray): Input features array of shape (n_samples, n )
            y (numpy.ndarray): Labels array of shape (n_samples, 1)
            iterations (int): Number of iterations for training.
            plot_cost (bool): Whether to plot cost over iterations or not.

        Returns:
            None.
        """
        pass

    def predictions(self, X):
        pass

    def save_model(self, filename=None):
        pass

    def load_model(cls, filename):
        pass
