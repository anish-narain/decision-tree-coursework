import numpy

from decision_tree import *

def confusion_matrix(true_values, predicted_values, num_classes):
    if len(true_values) != len(predicted_values):
        raise ValueError("Input lists must have the same length")

    matrix = np.zeros((num_classes, num_classes), dtype=int)

    '''
    row = true values
    column = predicted values
    '''
    
    for i in range(len(true_values)):
        true_class = int(true_values[i])
        predicted_class = int(predicted_values[i])
        matrix[true_class - 1, predicted_class - 1] += 1

    return matrix

def accuracy_from_confusion(confusion):
    if np.sum(confusion) > 0:
        return np.sum(np.diag(confusion)) / np.sum(confusion)
    else:
        return 0.

def precision_from_confusion(confusion):
    p = np.zeros((len(confusion), ))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[:, c]) > 0:
            p[c] = confusion[c, c] / np.sum(confusion[:, c])  
    macro_p = 0.
    if len(p) > 0:
        macro_p = np.mean(p)
    return p

def recall_from_confusion(confusion):
    r = np.zeros((len(confusion), ))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[c, :]) > 0:
            r[c] = confusion[c, c] / np.sum(confusion[c, :])
    return r

def f1_score_from_confusion(precisions, recalls):

    # just to make sure they are of the same length
    assert len(precisions) == len(recalls)

    f = np.zeros((len(precisions), ))
    for c, (p, r) in enumerate(zip(precisions, recalls)):
        if p + r > 0:
            f[c] = 2 * p * r / (p + r)
    return f


#Could build off of this ===============================================================
def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices

def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # pick k as test
        test_indices = split_indices[k]

        # combine remaining splits as train
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

        folds.append((train_indices, test_indices))

    return folds

def evaluate(test_db, trained_tree):
    predicted_values, true_values = [], []
    for test in test_db:
        predicted_values.append(trained_tree.make_prediction(test))
        true_values.append(test[-1])
    confusion = confusion_matrix(true_values, predicted_values, 4)
    confusion_mat = confusion
    acc = accuracy_from_confusion(confusion)
    prec = precision_from_confusion(confusion)
    rec = recall_from_confusion(confusion)
    f1 = f1_score_from_confusion(prec, rec)

    return acc, prec, rec, f1, confusion_mat


def cross_validation(database, random_generator=default_rng()):
    n_folds = 10
    train_test_folds =train_test_k_fold(n_folds, len(database), random_generator)

    acc = 0
    prec, rec, f1 = numpy.zeros((4, )), numpy.zeros((4,)), numpy.zeros((4,))
    confusion_mat = numpy.zeros((4, 4))

    for (train_indices, test_indices) in train_test_folds:
        # get the dataset from the correct splits
        database_train = database[train_indices, :]
        database_test = database[test_indices, :]
        trained_tree, depth = decision_tree_learning(database_train, depth=0)
        a, p, r, f, c = evaluate(database_test, trained_tree)
        acc += a
        prec += p
        rec += r
        f1 += f
        confusion_mat += c

    macro_prec = prec.mean() / n_folds
    macro_rec = rec.mean() / n_folds
    macro_f1 = f1.mean() / n_folds

    return acc / n_folds, macro_prec, macro_rec, macro_f1, confusion_mat / n_folds

