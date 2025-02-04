from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class SGPUncertainty:
    def __init__(self):
        self.kernel = RBF(length_scale=1.0)
        self.gp = GaussianProcessRegressor(kernel=self.kernel)

    def fit_predict(self, X, y):
        self.gp.fit(X, y)
        y_pred, sigma = self.gp.predict(X, return_std=True)
        return y_pred, sigma