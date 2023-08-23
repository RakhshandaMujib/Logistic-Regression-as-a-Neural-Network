import numpy as np

def confusion_mat(A, y):
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

def model_eval(A, y):
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
    tp, fp, tn, fn = confusion_mat(A, y)
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
