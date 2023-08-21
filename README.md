# Logistic-Regression-as-a-Neural-Network

This repository demonstrates binary classification using logistic regression (interpreted as a neural network). The goal is to predict binary outcomes (0 or 1) based on a set of features. </br>
The general notation and the math behind each operation are provided below.

### Dataset

- `X`: Set of features (m, n) where each element belongs to the set of real numbers.
- `y`: Set of ground truth labels (m, 1) where each element belongs to {0, 1}.

Where:
- `m`: Number of training examples
- `n`: Number of features

### Prediction

The predicted output `y_hat`, written as `a_i` for the `i`th data item, is calculated using the sigmoid function:

```
y_hat = sigmoid(Z)
```

Where:
- `Z = (W.T * X) + b` (m, 1)
- `W`: Set of weights (n, 1) initialized as 0
- `b`: Intercept (real number) initialized as 0
- `sigmoid = 1 / (1 + e^(-z))`

### Loss Function

The loss function penalizes the model for incorrect predictions and needs to be minimized:

```
Loss(a_i, y_i) = - [y_i * log(a_i) + (1 - y_i) * log(1 - a_i)]
```

- If `a_i > 0.5` and `y_i = 1`, the model is penalized for lack of confidence.
- If `a_i > 0.5` and `y_i = 0`, the model is penalized for high inaccuracy.

### Cost Function

The cost function is calculated for all training samples and needs to be minimized:

```
J(W, b) = (1 / m) * sum(Loss(A, y))
```

Where:
- `A`: Set of `m` predicted outputs `a_i`

### Update Weights and Bias

The weights and bias are updated using gradient descent:

```
W -= alpha * dW
b -= alpha * db
```

Where:
- `alpha`: Learning rate, determines the gradient descent speed
- `dW = (1 / m) * sum[(A - y) * X]`
- `db = (1 / m) * sum[(A - y)]`
- `dZ = A - y`

