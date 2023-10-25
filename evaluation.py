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
    
    return (p, macro_p)

def recall_from_confusion(confusion):
    r = np.zeros((len(confusion), ))
    for c in range(confusion.shape[0]):
        if np.sum(confusion[c, :]) > 0:
            r[c] = confusion[c, c] / np.sum(confusion[c, :])    

    # Compute the macro-averaged recall
    macro_r = 0.
    if len(r) > 0:
        macro_r = np.mean(r)
    
    return (r, macro_r)

def f1_score_from_confusion(confusion):
    (precisions, macro_p) = precision_from_confusion(confusion)
    (recalls, macro_r) = recall_from_confusion(confusion)

    # just to make sure they are of the same length
    assert len(precisions) == len(recalls)

    f = np.zeros((len(precisions), ))
    for c, (p, r) in enumerate(zip(precisions, recalls)):
        if p + r > 0:
            f[c] = 2 * p * r / (p + r)

    # Compute the macro-averaged F1
    macro_f = 0.
    if len(f) > 0:
        macro_f = np.mean(f)
    
    return (f, macro_f)


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

def evaluate(test_db, trained_tree, current_node):
    total_examples, correct_examples = 0, 0
    for test_instance in test_db:
        correct_examples += int(trained_tree.make_prediction(current_node, test_instance) == test_instance[-1])
        total_examples += 1
    accuracy = correct_examples / total_examples
    return accuracy

def cross_validation(database, random_generator=default_rng()):
    n_folds = 10
    train_test_folds =train_test_k_fold(n_folds, len(database), random_generator)
    
    err_sum = 0
    for (train_indices, test_indices) in train_test_folds:
        # get the dataset from the correct splits
        database_train = database[train_indices, :]
        database_test = database[test_indices, :]
        current_node, depth = decision_tree_learning(database_train, depth=0)
        trained_tree = DecisionTree(current_node.emitter,current_node.value)
        err_sum += evaluate(database_test, trained_tree, current_node)
    global_err_est = err_sum / n_folds
    return global_err_est
  



# def train_val_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
#     # split the dataset into k splits
#     split_indices = k_fold_split(n_folds, n_instances, random_generator)

#     folds = []
#     for k in range(n_folds):
#         # pick k as test, and k+1 as validation (or 0 if k is the final split)
#         test_indices = split_indices[k]
#         #val_indices = split_indices[(k+1) % n_folds]

#         # concatenate remaining splits for training
#         train_indices = np.zeros((0, ), dtype=np.int)
#         for i in range(n_folds):
#             # concatenate to training set if not validation or test
#             if i not in [k, (k+1) % n_folds]:
#                 train_indices = np.hstack([train_indices, split_indices[i]])

#         folds.append([train_indices, val_indices, test_indices])
        
#     return folds

