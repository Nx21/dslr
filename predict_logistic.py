import numpy  py as np
import math
def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	def sigmoid(z):
		return 1 / (1 + math.exp(-z))

	result = []
	for x in X:
		z = sum([w * i for w, i in zip(x, weights)]) + bias
		z = np.clip(z, -500, 500)
		result.append(1 if sigmoid(z) >= .5 else  0)
	return result