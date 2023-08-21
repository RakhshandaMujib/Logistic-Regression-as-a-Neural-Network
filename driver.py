from load_iris import *
from logistic_regression import *

def main():
  #Load the Iris Dataset:
  X_train, X_test, y_train, y_test = load_iris()
  model = LogisticRegression(X_train, y_train, iters = 1000, alpha = 0.005)
  model.fit(print_cost = True)
  y_pred = model.predict(X_test)
  model.model_eval(y_pred, y_test)
  model.plot_cost()

if __name__ == '__main__':
  main()
