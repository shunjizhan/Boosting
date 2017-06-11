import sys
import numpy as np
#from sklearn import metrics

def read_file(filename):
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

def getAlpha(e):
    return 0.5 * np.log((1 - e) / e)


# Main code starts here
T_max = int(sys.argv[1])
train_file_pos = sys.argv[2]
train_file_neg = sys.argv[3]
test_file_pos = sys.argv[4]
test_file_neg = sys.argv[5]

# Read data
train_pos = read_file(train_file_pos)
train_neg = read_file(train_file_neg)
test_pos = read_file(test_file_pos)
test_neg = read_file(test_file_neg)

# initialize weight
N = (train_pos.shape[0] + train_neg.shape[0])
weight = [1.0 / N] * N

# training process
x = 0
error_rate = 0
confidence = []
W_all = []
T_all = []
while(x < T_max and error_rate < 0.5):
    x += 1

    weight_pos = weight[0 : train_pos.shape[0]]
    weight_neg = weight[train_pos.shape[0] : ]

    # Centroids are just the mean of the NxD data along the first dimension (axis=0)
    centroid_A = np.average(train_pos, axis=0, weights = weight_pos)
    centroid_B = np.average(train_neg, axis=0, weights = weight_neg)

    # Lecture 4-5 slide 25: Compute 't' and 'w'
    W = (centroid_A - centroid_B)
    T = 0.5 * np.inner((centroid_A + centroid_B), (centroid_A - centroid_B))

    train_data = np.concatenate((train_pos, train_neg))
    Y_pred = []

    # For each train data point, predict class label by comparing X.W-T and store it in Y_pred
    for X in train_data:
        if (np.inner(X, W) - T) >= 0:
            class_pred = 1
        else:
            class_pred = 0

        Y_pred.append(class_pred)

    # Construct ground-truth label vector for the train data
    Y_true = np.concatenate((1*np.ones((train_pos.shape[0], 1)), 0*np.ones((train_pos.shape[0], 1))))    # Nx1 vector
    Y_pred = np.vstack(Y_pred)  # Convert from python list to Nx1 numpy vector

    # Compute confusion matrix (dimension 3x3) by comparing the ground-truth
    # label vector (Y_true) and the predicted label vector (Y_pred)
    conf_mat = confusion_matrix(Y_true, Y_pred) # aka contingency table

    # print np.asarray(Y_true).reshape(-1)
    # print np.asarray(Y_pred).reshape(-1)

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

    # compute the confidence value of current classifier
    alpha = getAlpha(error_rate)
    confidence.append(alpha)
    W_all.append(W)
    T_all.append(T)

    # recalculate the weights
    weight_new = []
    for j in range (0, len(Y_true)):
        if (Y_true[j] != Y_pred[j]):
            weight_new.append(weight[j] / (2 * error_rate))
        else:
            weight_new.append(weight[j] / (2 - (2 * error_rate)))

    weight = weight_new
    # print confidence
    # print sum(weight)
    # print weight


# testing process
test_data = np.concatenate((test_pos, test_neg))
Y_pred = []

# For each test data point, predict class label by comparing X.W-T and store it in Y_pred
for X in test_data:
    result = 0
    for y in range (0, len(W_all)):
        a = confidence[y]
        W = W_all[y]
        T = T_all[y]
        result += a * (np.inner(X, W) - T)

    if  result >= 0:
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

# print np.asarray(Y_true).reshape(-1)
# print np.asarray(Y_pred).reshape(-1)

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



# Print the results to console
print("True positive rate = %.2f" % tpr)
print("False positive rate = %.2f" % fpr)
print("Error rate = %.2f" % error_rate)
print("Accuracy = %.2f" % accuracy)
print("Precision = %.2f" % precision)
