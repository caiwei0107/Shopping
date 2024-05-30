import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import resample
from sklearn.neighbors import KDTree
from sklearn.metrics import confusion_matrix

TEST_SIZE = 0.4
# Set random seed for fixed result
np.random.seed(2)

# Nearest-neighbor classifier uses bagging
class KNN(BaseEstimator, ClassifierMixin):
    # Parameters n_estimators and max_samples are selected using grid search for optimal performance
    def __init__(self, k=1, n_estimators=3, max_samples=0.9):
        self.k = k
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.models = []

    def fit(self, X, y):
        self.models = []
        for _ in range(self.n_estimators):
            # Resample
            X_sample, y_sample = resample(X, y, replace=True, n_samples=int(self.max_samples * len(X)))
            model = MySimpleKNN(k=self.k)
            model.fit(X_sample, y_sample)
            self.models.append(model)

    def predict(self, X):
        # Collect predictions from each model
        predictions = np.array([model.predict(X) for model in self.models])
        # Return Majority vote
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

# A simplified version of a KNN model based on KDTree
class MySimpleKNN(BaseEstimator, ClassifierMixin):
    def __init__(self, k=1):
        self.k = k
        self.tree = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        self.tree = KDTree(self.X_train) # Euclidean distance

    def predict(self, X):
        X = np.array(X)
        dist, ind = self.tree.query(X, k=self.k)
        return np.array([np.bincount(self.y_train[indices]).argmax() for indices in ind])

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity , F1score = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")
    print(f"F1 Measure: {100 * F1score:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    data = pd.read_csv(filename)
    # pre-processing
    month_mapping = {'Jan':0,'Feb':1,'Mar':2,'Apr':3,'May':4,'June':5,'Jul':6,'Aug':7,'Sep':8,'Oct':9,'Nov':10,'Dec':11}
    data['Month'] = data['Month'].apply(lambda x:month_mapping[x])
    data['VisitorType'] = data['VisitorType'].apply(lambda x: 1 if x == 'Returning_Visitor' else 0)
    data['Weekend'] = data['Weekend'].apply(lambda x: 1 if x is True else 0)
    data['Revenue'] = data['Revenue'].apply(lambda x: 1 if x is True else 0)
    #print(data.head(1))
    return data[data.columns[:-1]],data['Revenue']
    raise NotImplementedError


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    MYknn = KNN()  # default k=1 ord=2

    # train
    MYknn.fit(evidence, labels)

    return MYknn
    raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    # confusion matrix
    cm = confusion_matrix(labels, predictions)
    #print(cm)
    TN, FP, FN, TP = cm.ravel()
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1_score_value = f1_score(labels, predictions, average='macro')
    return sensitivity, specificity , f1_score_value
    raise NotImplementedError


if __name__ == "__main__":
    main()
