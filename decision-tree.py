import numpy as np
import os
import matplotlib.pyplot as plt
import math
from numpy.random import default_rng

# def read_dataset(filepath):
#     x = []
#     y_labels = []
#     for line in open(filepath):
#         if line.strip(): 
#             row = line.strip().split("\t")
#             x.append(list(map(float, row[:-1])))
#             y_labels.append(row[-1])

#     [classes, y] = np.unique(y_labels, return_inverse=True)

#     x = np.array(x)
#     y = np.array(y)
#     return (x, y, classes) #return dataset

"""
Task 1
#(x, y, classes) = read_dataset("./wifi_db/clean_dataset.txt")
#print(x, "\n")
#print(y, "\n")
#print(classes, "\n")
"""

def split_training_testing_data(data_set, test_proportion, random_generator=default_rng()):
    instances, _ = data_set.shape
    test_size = round(test_proportion * instances)
    shuffled_indices = random_generator.permutation(instances)
    
    test = data_set[shuffled_indices[:test_size]]
    train = data_set[shuffled_indices[test_size:]]

    return tuple(map(np.array, (test, train)))

class Node:
    def __init__(self, emitter, value):
        self.emitter = emitter
        self.value = value
        self.room = None
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self, root_emitter, root_value):
        self.root = Node(root_emitter, root_value)
    
    def visualize_tree(self, node, level=0, prefix="Root: "):
        if node is not None:
            if node.left is None and node.right is None:
                print(" " * (level * 4), prefix, "leaf:", str(node.room))
            else:
                print(" " * (level * 4), prefix, ("[X" + str(node.emitter)), " < ", (str(node.value)+"]"))
            
            if node.left is not None or node.right is not None:
                self.visualize_tree(node.left, level + 1, "L--- ")
                self.visualize_tree(node.right, level + 1, "R--- ")

    def plot_tree(self, node, level=0, x=0, x_scale=1.0, y_scale=1.0):
        font_properties = {
            'family': 'serif',
            'color': 'black',
            'weight': 'normal',
        }
        fig, ax = plt.subplots(figsize=(20, 10))
        fig.patch.set_facecolor('white')

        def recursive_plot(node, level, x, x_scale, y_scale):
            if node is not None:
                if node.left is None and node.right is None:
                    # This is a leaf node
                    node_text = f"leaf:{node.room}"
                    ax.text(x, -level * y_scale, node_text, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'), fontdict=font_properties)
                else:
                    # This is not a leaf node
                    node_text = f"X{node.emitter} < {node.value}"
                    ax.text(x, -level * y_scale, node_text, ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'), fontdict=font_properties)
                
                
                # Recursively plot left and right subtrees
                if node.left is not None:
                    x_left = x - 1.0 / (2 ** level) * x_scale
                    plt.plot([x, x_left], [-level * y_scale, -level * y_scale - 1], 'ro-', linewidth=2, markersize=8)
                    recursive_plot(node.left, level + 1, x_left, x_scale, y_scale)
                if node.right is not None:
                    x_right = x + 1.0 / (2 ** level) * x_scale
                    plt.plot([x, x_right], [-level * y_scale, -level * y_scale - 1], 'bo-', linewidth=2, markersize=8)
                    recursive_plot(node.right, level + 1, x_right, x_scale, y_scale)
        
        recursive_plot(node, level, x, x_scale, y_scale)
        
        ax.axis('off')
        #plt.title('Decision Tree Visualization', fontsize=16, fontdict=font_properties)
        plt.show()
    
    def make_prediction(self, node, testing_instance):
        current_node = node
        while (current_node.left or current_node.right):
            emitter_value = testing_instance[current_node.emitter]
            if emitter_value < current_node.value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return current_node.room
    
def find_split(dataset):
    max_info_gain, feature, split = 0, 0, 0
    num_instances, num_features = dataset.shape
    for f in range(num_features - 1):
        sorted_indices = np.argsort(dataset[:, f])
        sorted_dataset = dataset[sorted_indices]
        for instance in range(num_instances-1):
            if sorted_dataset[instance][f] != sorted_dataset[instance+1][f]:
                s_left = sorted_dataset[:instance+1]
                s_right = sorted_dataset[instance+1:]
                gain = calculate_information_gain(sorted_dataset, s_left, s_right)
                if gain > max_info_gain:
                    max_info_gain = gain
                    feature = f
                    split = sorted_dataset[instance+1][f]
    return feature, split


def calculate_entropy(data_set):
    _, counts = np.unique(data_set[:, -1], return_counts=True)
    total = sum(counts)
    entropy = sum((-1) * math.log2(count / total) * (count / total) for count in counts)
    return entropy


def calculate_information_gain(s_all, s_left, s_right):
    remainder = (len(s_left) * calculate_entropy(s_left) / len(s_all))  + (len(s_right) * calculate_entropy(s_right)/ len(s_all))
    info_gain = calculate_entropy(s_all) - remainder
    return info_gain

def split_dataset(training_dataset, split_emitter, split_value):
    # Extract the emitter column
    emitter_column = training_dataset[:, split_emitter]

    # Create boolean masks for the left and right datasets
    left_mask = emitter_column < split_value
    right_mask = emitter_column >= split_value

    # Use boolean emittering to create the left and right datasets
    left_dataset = training_dataset[left_mask]
    right_dataset = training_dataset[right_mask]

    return left_dataset, right_dataset

def reached_leaf(training_dataset):
    return np.all(training_dataset[:, -1] == training_dataset[:, -1][0])

def decision_tree_learning(training_dataset, depth):
    if reached_leaf(training_dataset):
        class_column = training_dataset[:, -1]
        leaf_label = class_column[0]
        leaf_node = Node("leaf", 0.000)
        leaf_node.room = leaf_label
        return leaf_node, depth
    else:
        split_emitter, split_value = find_split(training_dataset)
        
        if split_value == 0.0:
            # If the split value is 0, create a leaf node with the majority class
            elements, count = np.unique(training_dataset[:, -1], return_counts=True)
            majority_class = elements[np.argmax(count)]
            leaf_node = Node("leaf", 0.000)
            leaf_node.room = majority_class
            return leaf_node, depth

        tree_pointer = Node(split_emitter, split_value)

        left_dataset, right_dataset = split_dataset(training_dataset, split_emitter, split_value)
        tree_pointer.left, left_depth = decision_tree_learning(left_dataset, depth+1)
        tree_pointer.right, right_depth = decision_tree_learning(right_dataset, depth+1)
        
        return tree_pointer, max(left_depth, right_depth)


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


def train_val_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # pick k as test, and k+1 as validation (or 0 if k is the final split)
        test_indices = split_indices[k]
        val_indices = split_indices[(k+1) % n_folds]

        # concatenate remaining splits for training
        train_indices = np.zeros((0, ), dtype=np.int)
        for i in range(n_folds):
            # concatenate to training set if not validation or test
            if i not in [k, (k+1) % n_folds]:
                train_indices = np.hstack([train_indices, split_indices[i]])

        folds.append([train_indices, val_indices, test_indices])
        
    return folds

def main():
    dataset = np.loadtxt("wifi_db/clean_dataset.txt")
    seed = 60012
    rg = default_rng(seed)
    testing_dataset, training_dataset = split_training_testing_data(dataset, 0.2, rg)
 
    # Call the decision_tree_learning function to build the decision tree
    tree, depth = decision_tree_learning(training_dataset, depth=0)

    current_node = tree
   
    new_tree = DecisionTree(current_node.emitter,current_node.value)
    print(new_tree.visualize_tree(current_node))
    new_tree.plot_tree(current_node)
    
    count, total = 0, 0

    true_dataset = []
    prediction_dataset = []

    for instance in testing_dataset:
        predicted_value = new_tree.make_prediction(current_node, instance)
        count += int(predicted_value == instance[-1])
        total += 1
        true_dataset.append(instance[-1])
        prediction_dataset.append(predicted_value)
        
    print("Accuracy: ", count / total)

    true_dataset1 = [1.0, 3.0, 2.0, 1.0, 4.0]
    prediction_dataset1 = [3.0, 2.0, 3.0, 1.0, 3.0]

    confusionmatrix = confusion_matrix(true_dataset,prediction_dataset, 4)
    print(confusionmatrix)
    print(accuracy_from_confusion(confusionmatrix))
    print(precision_from_confusion(confusionmatrix))
    print(recall_from_confusion(confusionmatrix))
    print(f1_score_from_confusion(confusionmatrix))
    
    # Print the final leaf node
    #print(f"Leaf Node value: {current_node.value}")

if __name__ == "__main__":
    main()