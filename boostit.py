import sys
import numpy as np
#from sklearn import metrics

def read_file(filename):
    """
    Reads a file and returns the data as a numpy array for the three classes
    """
    return np.loadtxt(filename, skiprows=1)


def confusion_matrix(Y_true, Y_pred):
    """
    Outputs a confusion matrix by comparing true labels and predicted labels
    """
    num_classes = len(np.unique(Y_true))
    conf_mat = np.zeros((num_classes, num_classes))

    for y_t, y_p in zip(Y_true, Y_pred):
        conf_mat[int(y_p)][int(y_t)] += 1

    return conf_mat


def classifier_metrics(conf_mat, curr_class):
    """
    Computes true positive rate, false positive rate, accuracy, error rate and precision
    for a given class from a confusion matrix.
    """
    not_curr_class = [c for c in range(conf_mat.shape[0]) if c != curr_class]

    tp = conf_mat[curr_class, curr_class]
    fp = conf_mat[curr_class, not_curr_class].sum()
    fn = conf_mat[not_curr_class, curr_class].sum()
    tn = conf_mat[np.ix_(not_curr_class, not_curr_class)].sum()

    tpr = tp / (tp + fn)
    fpr = fp / (fp + tn)
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    error_rate = 1 - accuracy
    precision = tp / (tp + fp)

    return tpr, fpr, accuracy, error_rate, precision


# Main code starts here
T = sys.argv[1]
train_file_pos = sys.argv[2]
train_file_neg = sys.argv[3]
test_file_pos = sys.argv[4]
test_file_neg = sys.argv[5]

# Read training data
train_pos = read_file(train_file_pos)
train_neg = read_file(train_file_neg)
test_pos = read_file(test_file_pos)
test_neg = read_file(test_file_neg)

# Centroids are just the mean of the NxD data along the first dimension (axis=0)
centroid_A = np.mean(train_pos, axis=0)
centroid_B = np.mean(train_neg, axis=0)

# Lecture 4-5 slide 25: Compute 't' and 'w'
W = (centroid_A - centroid_B)
T = 0.5 * np.inner((centroid_A + centroid_B), (centroid_A - centroid_B))

test_data = np.concatenate((test_pos, test_neg))
Y_pred = []

# For each test data point, predict class label by comparing X.W-T and store it in Y_pred
for X in test_data:
    if (np.inner(X, W) - T) >= 0:
        class_pred = 1
    else:
        class_pred = 0

    Y_pred.append(class_pred)

# Construct ground-truth label vector for the test data
Y_true = np.concatenate((1*np.ones((test_pos.shape[0], 1)), 0*np.ones((test_pos.shape[0], 1))))    # Nx1 vector
Y_pred = np.vstack(Y_pred)  # Convert from python list to Nx1 numpy vector

# Compute confusion matrix (dimension 3x3) by comparing the ground-truth
# label vector (Y_true) and the predicted label vector (Y_pred)
conf_mat = confusion_matrix(Y_true, Y_pred) # aka contingency table

# scikit-learn's in-built function can also compute the confusion matrix for us:
# conf_mat = metrics.confusion_matrix(Y_true, Y_pred)

tpr_A, fpr_A, accuracy_A, error_A, precision_A = classifier_metrics(conf_mat, 1)
tpr_B, fpr_B, accuracy_B, error_B, precision_B = classifier_metrics(conf_mat, 0)

# Compute metrics across the three classes
tpr = (tpr_A + tpr_B) / 2
fpr = (fpr_A + fpr_B) / 2
precision = (precision_A + precision_B) / 2
# Note that the 3-class classifier's accuracy is not just the average of individual class accuracies!
accuracy = conf_mat.trace() / conf_mat.sum()    # trace = sum of diagonal values = sum of TPs
error_rate = 1 - accuracy

# The above values can also be computed very easily via sklearn's metrics package (from sklearn import metrics)

# tpr1 = metrics.recall_score(Y_true, Y_pred, average='macro')
# precision1 = metrics.precision_score(Y_true, Y_pred, average='macro')
# accuracy1 = metrics.accuracy_score(Y_true, Y_pred, normalize=True)
# error_rate1 = 1 - accuracy

# Print the results to console
print("True positive rate = %.2f" % tpr)
print("False positive rate = %.2f" % fpr)
print("Error rate = %.2f" % error_rate)
print("Accuracy = %.2f" % accuracy)
print("Precision = %.2f" % precision)
