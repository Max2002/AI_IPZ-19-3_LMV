import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, \
    roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv('data_metrics.csv')
df.head()

thresh = 0.5
df['predicted_RF'] = (df.model_RF >= thresh).astype('int')
df['predicted_LR'] = (df.model_LR >= thresh).astype('int')
df.head()

# confusion_matix

print(confusion_matrix(df.actual_label.values, df.predicted_RF.values))


def find_TP(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 1))


def find_FN(y_true, y_pred):
    return sum((y_true == 1) & (y_pred == 0))


def find_FP(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 1))


def find_TN(y_true, y_pred):
    return sum((y_true == 0) & (y_pred == 0))


print('TP:', find_TP(df.actual_label.values, df.predicted_RF.values))
print('FN:', find_FN(df.actual_label.values, df.predicted_RF.values))
print('FP:', find_FP(df.actual_label.values, df.predicted_RF.values))
print('TN:', find_TN(df.actual_label.values, df.predicted_RF.values))


def find_conf_matrix_values(y_true, y_pred):
    TP = find_TP(y_true, y_pred)
    FN = find_FN(y_true, y_pred)
    FP = find_FP(y_true, y_pred)
    TN = find_TN(y_true, y_pred)
    return TP, FN, FP, TN


def liashuk_confusion_matrix(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return np.array([[TN, FP], [FN, TP]])


print(liashuk_confusion_matrix(df.actual_label.values, df.predicted_RF.values))

assert np.array_equal(liashuk_confusion_matrix(df.actual_label.values, df.predicted_RF.values),
                      confusion_matrix(df.actual_label.values,
                                       df.predicted_RF.values)), 'my_confusion_matrix() is not correct for RF'
assert np.array_equal(liashuk_confusion_matrix(df.actual_label.values, df.predicted_LR.values),
                      confusion_matrix(df.actual_label.values,
                                       df.predicted_LR.values)), 'my_confusion_matrix() is not correct for LR'


# accuracy
score = accuracy_score(df.actual_label.values, df.predicted_RF.values)
print("Accuracy score on RF:", score)


def liashuk_accuracy_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return (TP + TN) / (TP + FN + FP + TN)


assert liashuk_accuracy_score(df.actual_label.values, df.predicted_RF.values) == accuracy_score(df.actual_label.values, df.predicted_RF.values), \
    'my accuracy_score failed on RF'

assert liashuk_accuracy_score(df.actual_label.values, df.predicted_LR.values) == accuracy_score(df.actual_label.values, df.predicted_LR.values), \
    'my accuracy_score failed on LR'

print("my accuracy score on RF:", liashuk_accuracy_score(df.actual_label.values, df.predicted_RF.values))
print("my accuracy score on LR:", liashuk_accuracy_score(df.actual_label.values, df.predicted_LR.values))

# Recall
print('Recall score on RF:', recall_score(df.actual_label.values, df.predicted_RF.values))


def liashuk_recal_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FN)


assert liashuk_recal_score(df.actual_label.values, df.predicted_RF.values) == recall_score(df.actual_label.values, df.predicted_RF.values),\
    'my recal_score fails on RF'

assert liashuk_recal_score(df.actual_label.values, df.predicted_LR.values) == recall_score(df.actual_label.values, df.predicted_LR.values),\
    'my recal_score fails on LR'

print("My recall score on RF:", liashuk_recal_score(df.actual_label.values, df.predicted_RF.values))
print("My recall score on LR:", liashuk_recal_score(df.actual_label.values, df.predicted_LR.values))

# precision_score

print("Precision score on RF:", precision_score(df.actual_label.values, df.predicted_RF.values))

def liashuk_precision_score(y_true, y_pred):
    TP, FN, FP, TN = find_conf_matrix_values(y_true, y_pred)
    return TP / (TP + FP)


assert liashuk_precision_score(df.actual_label.values, df.predicted_RF.values) == precision_score(df.actual_label.values, df.predicted_RF.values),\
    'my precision_score fails on RF'

assert liashuk_precision_score(df.actual_label.values, df.predicted_LR.values) == precision_score(df.actual_label.values, df.predicted_LR.values),\
    'my precision_score fails on LR'

print("my precision score on RF:", liashuk_precision_score(df.actual_label.values, df.predicted_RF.values))
print("my precision score on LR:", liashuk_precision_score(df.actual_label.values, df.predicted_LR.values))

# F1 score
print("F1 score on RF", f1_score(df.actual_label.values, df.predicted_RF.values))
def liashuk_f1_score(y_true, y_pred):
    precision = liashuk_precision_score(y_true, y_pred)
    recall = liashuk_recal_score(y_true, y_pred)
    return (2 * (precision * recall)) / (precision + recall)


assert liashuk_f1_score(df.actual_label.values, df.predicted_RF.values) == f1_score(df.actual_label.values, df.predicted_RF.values),\
    'my f1_score fails on RF'

assert liashuk_f1_score(df.actual_label.values, df.predicted_LR.values) == f1_score(df.actual_label.values, df.predicted_LR.values),\
    'my f1_score fails on LR'

print("My F1 score score on RF:", liashuk_f1_score(df.actual_label.values, df.predicted_RF.values))
print("My F1 score score on LR:", liashuk_f1_score(df.actual_label.values, df.predicted_LR.values))
print()


def test_thresholds(threshold: float):
    print(f"Scores with threshold = {threshold}")
    predicted = (df.model_RF >= threshold).astype('int')

    print("Accuracy RF:", liashuk_accuracy_score(df.actual_label.values, predicted))
    print("Precision RF:", liashuk_precision_score(df.actual_label.values, predicted))
    print("Recall RF:", liashuk_recal_score(df.actual_label.values, predicted))
    print("F1 RF:", liashuk_f1_score(df.actual_label.values, predicted))
    print()


test_thresholds(thresh)
test_thresholds(.25)
test_thresholds(.75)
test_thresholds(.15)

# roc curve
fpr_RF, tpr_RF, thresholds_RF = roc_curve(df.actual_label.values, df.model_RF.values)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(df.actual_label.values, df.model_LR.values)

# roc auc score
auc_RF = roc_auc_score(df.actual_label.values, df.model_RF.values)
auc_LR = roc_auc_score(df.actual_label.values, df.model_LR.values)

print("AUC RF:", auc_RF)
print("AUC LR:", auc_LR)

plt.plot(fpr_RF, tpr_RF, 'r-', label=f'AUC RF: {auc_RF}')
plt.plot(fpr_LR, tpr_LR, 'b-', label=f'AUC LR: {auc_LR}')
plt.plot([0, 1], [0, 1], 'k-', label='random')
plt.plot([0, 0, 1, 1], [0, 1, 1, 1], 'g-', label='perfect')

plt.legend()

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.show()