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
    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value
        self.left = None
        self.right = None


class DecisionTree:
    def __init__(self, root_attribute, root_value):
        self.root = Node(root_attribute, root_value)
    
    def visualize_tree(self, node, level=0, prefix="Root: "):
        if node is not None:
            print(" " * (level * 4), prefix, "Wifi Signal: ", str(node.attribute), " < ", str(node.value))
            if node.left is not None or node.right is not None:
                self.visualize_tree(node.left, level + 1, "L--- ")
                self.visualize_tree(node.right, level + 1, "R--- ")

    def plot_tree(self, node, level=0, x=0, x_scale=1.0, y_scale=1.0):
        # Create a single plot outside of the recursive function
        fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the figure size as needed
        
        def recursive_plot(node, level, x, x_scale, y_scale):
            if node is not None:
                # Draw the current node with styling
                node_text = f"{node.attribute} < {node.value}"
                ax.text(x, -level * y_scale, node_text, ha='center', va='center', bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.2'))
                
                # Recursively plot left and right subtrees
                if node.left is not None:
                    x_left = x - 1.0 / (2 ** level) * x_scale
                    plt.plot([x, x_left], [-level * y_scale, -level * y_scale - 1], 'ro-', linewidth=2, markersize=8)  # Draw a red line to left child
                    recursive_plot(node.left, level + 1, x_left, x_scale, y_scale)
                if node.right is not None:
                    x_right = x + 1.0 / (2 ** level) * x_scale
                    plt.plot([x, x_right], [-level * y_scale, -level * y_scale - 1], 'bo-', linewidth=2, markersize=8)  # Draw a blue line to right child
                    recursive_plot(node.right, level + 1, x_right, x_scale, y_scale)
        
        recursive_plot(node, level, x, x_scale, y_scale)
        
        # Turn off the axis and set the plot title
        ax.axis('off')
        plt.title('Decision Tree Visualization', fontsize=16)
        
        # Display the plot
        plt.show()
    
    def make_prediction(self, node, testing_instance):
        current_node = node
        while (current_node.left or current_node.right):
            attribute_value = testing_instance[current_node.attribute]
            if attribute_value < current_node.value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return current_node.value
    

        

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
    print(feature, split, '\n')
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

def split_dataset(training_dataset, split_attribute, split_value):
    # Extract the attribute column
    attribute_column = training_dataset[:, split_attribute]

    # Create boolean masks for the left and right datasets
    left_mask = attribute_column < split_value
    right_mask = attribute_column >= split_value

    # Use boolean indexing to create the left and right datasets
    left_dataset = training_dataset[left_mask]
    right_dataset = training_dataset[right_mask]

    return left_dataset, right_dataset

def reachedLeaf(training_dataset):
    return np.all(training_dataset[:, -1] == training_dataset[:, -1][0])


def decision_tree_learning(training_dataset, depth):
    if reachedLeaf(training_dataset):
        class_column = training_dataset[:, -1]
        leaf_label = class_column[0]
        # Create a leaf node and return it
        return Node(leaf_label, 0.0000), depth
    else:
        split_attribute, split_value = find_split(training_dataset)
        
        if split_value == 0.0:
            # If the split value is 0, create a leaf node with the majority class
            elements, count = np.unique(training_dataset[:, -1], return_counts=True)
            majority_class = elements[np.argmax(count)]
            return Node("Leaf", majority_class), depth

        tree_pointer = Node(split_attribute, split_value)

        left_dataset, right_dataset = split_dataset(training_dataset, split_attribute, split_value)
        tree_pointer.left, left_depth = decision_tree_learning(left_dataset, depth+1)
        tree_pointer.right, right_depth = decision_tree_learning(right_dataset, depth+1)
        
        return tree_pointer, max(left_depth, right_depth)
   

def main():
    dataset = np.loadtxt("wifi_db/clean_dataset.txt")
    seed = 60012
    rg = default_rng(seed)
    testing_dataset, training_dataset = split_training_testing_data(dataset, 0.2, rg)
 
    # Call the decision_tree_learning function to build the decision tree
    tree, depth = decision_tree_learning(training_dataset, depth=0)

    # You can now traverse the tree and make predictions or explore its structure
    # For example, let's traverse the leftmost branch until a leaf is reached:
    current_node = tree
   
    new_tree = DecisionTree(current_node.attribute,current_node.value)
    print(new_tree.visualize_tree(current_node))
    print(new_tree.plot_tree(current_node))
    for instance in testing_dataset:
        predicted_value = new_tree.make_prediction(current_node, instance)
        print(predicted_value, instance[-1])


    # Print the final leaf node
    #print(f"Leaf Node Value: {current_node.value}")


if __name__ == "__main__":
    main()