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

class TreeNode:
    def __init__(self, emitter, value):
        self.emitter = emitter
        self.value = value
        self.room = None
        self.left = None
        self.right = None
    
    def visualize_tree(self, level=0, prefix="Root: "):

        if self.left is None and self.right is None:
            print(" " * (level * 4), prefix, "leaf:", str(self.room))
        else:
            print(" " * (level * 4), prefix, ("[X" + str(self.emitter)), " < ", (str(self.value)+"]"))

        if self.left is not None or self.right is not None:
            self.left.visualize_tree(level + 1, "L--- ")
            self.right.visualize_tree( level + 1, "R--- ")

    def plot_tree(self, level=0, x=0, x_scale=1.0, y_scale=1.0):
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
        
        recursive_plot(self, level, x, x_scale, y_scale)
        
        ax.axis('off')
        #plt.title('Decision Tree Visualization', fontsize=16, fontdict=font_properties)
        plt.show()
    
    def make_prediction(self, testing_instance):
        current_node = self
        while (current_node.left or current_node.right):
            emitter_value = testing_instance[self.emitter]
            if emitter_value < self.value:
                current_node = self.left
            else:
                current_node = self.right
        return self.room
    
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
        leaf_node = TreeNode("leaf", 0.000)
        leaf_node.room = leaf_label
        return leaf_node, depth
    else:
        split_emitter, split_value = find_split(training_dataset)
        
        if split_value == 0.0:
            # If the split value is 0, create a leaf node with the majority class
            elements, count = np.unique(training_dataset[:, -1], return_counts=True)
            majority_class = elements[np.argmax(count)]
            leaf_node = TreeNode("leaf", 0.000)
            leaf_node.room = majority_class
            return leaf_node, depth

        tree_pointer = TreeNode(split_emitter, split_value)

        left_dataset, right_dataset = split_dataset(training_dataset, split_emitter, split_value)
        tree_pointer.left, left_depth = decision_tree_learning(left_dataset, depth+1)
        tree_pointer.right, right_depth = decision_tree_learning(right_dataset, depth+1)
        
        return tree_pointer, max(left_depth, right_depth)


