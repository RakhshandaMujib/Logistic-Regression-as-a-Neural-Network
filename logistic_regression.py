import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    
    def __init__(self, X, y, iters = 1000, thresh = 0.0005, alpha = 0.01):
        '''
        Brief: Creates the classifier object for logistic regression.
        
        Argument(s):
        * X - np.ndarray of shape  mxn. Features of the training dataset. 
        * y - np.ndarray of shape mx1. Ground truth labels of the training
            dataset.
        * iters - int, no. of training iterations. Default set to 1000.
        * thresh - int, threshold value for early stopping the training. 
            Default set to 0.0005.
        * alpha - float, learning rate for the gradient descent. Default
            set to 0.01.
        '''
        #Transposing the original data to match the shapes with the weight 
        #matrix and make other computations convenient
        self.X = X.T  #Shape becomes nxm
        self.y = y.T  #Shape becomes 1xm - row vector
    
        self.n, self.m = self.X.shape  #n = #features
                                       #m = #samples
            
        #Initialize the weight matrix as a column vector of nx1 
        #dimension with 0s.
        self.W = np.zeros(self.n).reshape(self.n, 1)
        
        self.b = 0  #Initialize the bias with 0
        
        #Initialize the hyperparameters:
        self.iters = iters  
        self.thresh = thresh 
        self.alpha = alpha  
        
        self.costs = [] #Costs for checking if to early stop or not
        
    
    def sigmoid(self, z):
        '''
        Brief: Computes the probabilities of the predicted output. The sigmoid 
        function ensures the probabilities lie within the range (0,1). 
        
        Argument(s): 
        * z - np.array, vector for which the sigmoid needs to be computed. Has
            a shape m x 1.

        Returns:
        np.array, sigmoid(z). Has shape m x 1.
        '''
        return 1 / (1 + np.exp(np.negative(z)))
    

    def loss(self, A, y):
        '''
        Brief: Computes the loss of the model for each predicted output using 
        the following formula:
            L(a_i, y_i) = - ((y_i * log(a_i)) + ((1 - y_i) * log(1 - a_i)))

        Argument(s):
        * A - np.ndarray, set of predicted outputs - a_i.
        * y - np.array, set of groud truth labels - y_i. Has shape 1 x m. 

        Returns:
        np.array, loss for each training example. Has shape m x 1.
        '''

        left_term = y * np.log(A)
        right_term = (1 - y) * np.log(1 - A)

        return np.negative(left_term + right_term)
    

    def cost(self, L):
        '''
        Brief: Computes the cost of the model by averaging the losses over the 
        entire training dataset. The formula used is:
            J(W, b) = sum( L(a_i, y_i)) / m

        Argument(s):
        * L - np.array, loss obtained for each training sample. Has shape m x 1.

        Returns:
        float, cost of the model.
        '''
        return np.mean(L)


    def predict(self, X, training = False):
        '''
        Brief: Computes the predicted probabilities / labels given the training/
        test set using the formula:
            a_i = sigmoid((W.T . X) + B)
        For the test set, if a_i > 0.5 the label is set to 1
                             a_i < 0.5 the label is set to 0

        Argument(s):
        * X - np.ndarray, dataset for prediction of the probabilities or the 
            class labels. Has shape  n x m, in case of training samples or 
            m_test x n, where m_test are the number of test samples.
        * training - bool, flag variable. If set to true computes the 
            probabilities else, the labels.

        Returns:
        np.array, vector of probabilities (shape  m x 1) or labels (m_test x 1)
        for training or test set. 
        '''
        #Change the shape of X to match the shape of W if X is the test set.
        X = X if training else X.T 

        Z = self.W.T @ X + self.b
        A = self.sigmoid(Z)
        labels = np.where(A < 0.5, 0, 1)

        if training:
            return A
        return labels.T


    def forward_pass(self):
        '''
        Brief: Propagates through the training set to compute the probabilities.

        Returns:
        tuple of computed probabilities for each training sample and their 
        corresponding losses.
        '''
        A = self.predict(self.X, True) #Probabilities. Has shape m x 1.
        L = self.loss(A, self.y) #Loss. Has shape m x 1.
        return A, L
        

    def backward_pass(self, A, L):
        '''
        Brief: Computes the gradients of the weight matrix and bias using the
        formula:
            d(W) / dJ = (1 / m) * ((A - y) . X)
            d(b) / dJ = (1 / m) *(A - y) 
        and performs the appropriate weight and bias updations using:
            W = W - alpha * (d(W) / dJ)
            b = b - alpha * (d(b) / dJ)

        Argument(s):
        * A - np.array, vector of predicted probabilities. Has shape m x 1.
        * L - np.array, vector of computed losses. Has shape m x 1.

        Returns:
        float, cost of the model.
        '''

        J = self.cost(L) #Cost of the model.

        #Keep a track of the costs for early stopping
        self.costs.append(J)

        #Compute the gradients:
        dW = (self.X @ (A - self.y).T) / self.m
        db = np.sum(A - self.y) / self.m
        #Update the wights and bias - gradient descent:
        self.W -= self.alpha * dW #Weights of the model. Has shape n x 1.
        self.b -= self.alpha * db #Bias of the model. 
        
        return J
        

    def fit(self, print_cost = False):
        '''
        Brief: Compiles the forward and the backward passes to train the model.
        '''

        old_cost = float('-inf') #Initialize the old cost.
        
        #Carry out the training iters number of iterations:
        for i in range(self.iters):

            #Forward pass -> Get the predicted values and loss.
            predicted_values, loss = self.forward_pass() 

            #Backward pass -> Compute gradients, update weights & bias, get the
            #model's cost.
            curr_cost = self.backward_pass(predicted_values, loss)
            
            if print_cost:
                print(f"Running pass {i + 1}...\n\tCost = {curr_cost}")

            #Stop the if the early stopping threshold is hit.
            if np.abs(old_cost - curr_cost) < self.thresh:
                print(f"Early stopping at pass {i+1}...")
                break
            old_cost = curr_cost #Update the old cost.

        print('Training successful!')
    

    def confusion_mat(self, A, y):
        '''
        Brief: Computes the confusion matrix for binary classification on the
        test set. 
                                            Actual
                                    ____________________________
                                   |  Positive (1) | Negative (0)
          Predicted  | Positive (1)|     TP        |     FP
                     | Negative (0)|     FN        |     TN

        Argument(s):
        * A - np.array, vector of predicted labels. Has shape m_test x 1.
        * y - np.array, set of groud truth labels - y_i. Has shape m_test x 1. 
        
        Returns:
        tuple of integers in the order TP, Fp, TN, FN.
        '''
        tp = np.sum((A == 1) & (y == 1))
        fp = np.sum((A == 1) & (y == 0))
        tn = np.sum((A == 0) & (y == 0))
        fn = np.sum((A == 0) & (y == 1))
        return tp, fp, tn, fn

    def model_eval(self, A, y):
        '''
        Brief: Evaluates the given model on the following metrics for binary
        classification.
            Metrics:
        - Accuracy: Percentage of correct predictions.
        - Precision: Proportion of true positive predictions among all 
          positive predictions.
        - Recall (Sensitivity): Proportion of true positive predictions
          among all actual positives.
        - F1 Score: Harmonic mean of precision and recall.
        - Specificity: Proportion of true negative predictions among all
          actual negatives.
        
        Argument(s):
        * A - np.array, vector of predicted labels. Has shape m_test x 1.
        * y - np.array, set of groud truth labels - y_i. Has shape m_test x 1. 

        '''
        if A.shape != y.shape:
            raise ValueError("Arrays A and y must have the same shape for elementwise comparison.")

        acc = np.mean(A == y) * 100
        tp, fp, tn, fn = self.confusion_mat(A, y)
        precision = tp / (tp + fp)
        recall = tp / (tp + tn)
        f1_score = (2 * precision * recall) / (precision + recall)
        specificity = tn / (fp + tn)
        
        print("*" * 10 + " Model report " + "*" * 10)
        print(f"   Accuracy = {acc:.2f}%")
        print(f"   Precision = {precision:.2f}")
        print(f"   Recall (Sensitivity) = {recall:.2f}")
        print(f"   F1 Score = {f1_score:.2f}")
        print(f"   Specificity = {specificity:.2f}")
        print()
        print(f"   True Positives (TP) = {tp}")
        print(f"   False Positives (FP) = {fp}")
        print(f"   True Negatives (TN) = {tn}")
        print(f"   False Negatives (FN) = {fn}")
        print("  " + "-" * 27)
        print(f"   Total samples = {y.shape[0]}")
        

    def plot_cost(self):
        '''
        Brief: Plots the iteration vs cost graph for the given model.
        '''
        plt.plot(self.costs)
        plt.xlabel("Number of iterations")
        plt.ylabel("J(W,b)")
        plt.title(f"Iterations vs Cost at alpha = {self.alpha}")
        plt.show()
