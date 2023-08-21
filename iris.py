import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_iris():
  iris_data = pd.read_csv('https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv')
  
  #Select two features and the target variable
  X = iris_data[['sepal_width', 'petal_width', 'sepal_length', 'petal_length']]
  y = (iris_data['species'] == 'virginica').astype(int)

  #Split the data into training and testing sets
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  X_train = np.array(X_train)
  y_train = np.array(y_train).reshape(y_train.shape[0], 1)
  X_test = np.array(X_test)
  y_test = np.array(y_test).reshape(y_test.shape[0], 1)

return X_train, X_test, y_train, y_test
