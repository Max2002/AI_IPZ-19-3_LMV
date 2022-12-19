import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

input_file = 'data_multivar_nb.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
classifier = svm.SVC(decision_function_shape='ovr')
classifier.fit(X_train, y_train)
y_test_pred = classifier.predict(X_test)

visualize_classifier(classifier, X_test, y_test)

num_folds = 3
accuracy_values = cross_val_score(classifier, X_test, y_test, scoring='accuracy', cv=num_folds)
print(f"Accuracy: {round(100 * accuracy_values.mean(), 2)}%")

precision_values = cross_val_score(classifier, X_test, y_test, scoring='precision_weighted', cv=num_folds)
print(f"Precision: {round(100 * precision_values.mean(), 2)}%")

recall_values = cross_val_score(classifier, X_test, y_test, scoring='recall_weighted', cv=num_folds)
print(f"Recall: {round(100 * recall_values.mean(), 2)}%")

f1_values = cross_val_score(classifier, X_test, y_test, scoring='f1_weighted', cv=num_folds)
print(f"F1: {round(100 * f1_values.mean(), 2)}%")