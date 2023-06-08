
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class DataExaminer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.is_binary_classification = self.check_binary_classification()
        self.is_linearly_separable = self.check_linear_separability()

    def check_binary_classification(self):
        return len(np.unique(self.y)) == 2

    def check_linear_separability(self):
        if not self.is_binary_classification:
            return False
        
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        return acc > 0.7
