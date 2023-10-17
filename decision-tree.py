import numpy as np
import os
import matplotlib

def read_dataset(filepath):
    x = []
    y_labels = []
    for line in open(filepath):
        if line.strip(): 
            row = line.strip().split("\t")
            x.append(list(map(float, row[:-1])))
            y_labels.append(row[-1])

    [classes, y] = np.unique(y_labels, return_inverse=True)

    x = np.array(x)
    y = np.array(y)
    return (x, y, classes) #return dataset

"""
Task 1
#(x, y, classes) = read_dataset("./wifi_db/clean_dataset.txt")
#print(x, "\n")
#print(y, "\n")
#print(classes, "\n")
"""
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

def find_split(dataset):
    ...

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
        # Create a leaf node and return it
        return Node("Leaf", 0.0000), depth
    else:
        split_attribute, split_value = find_split(training_dataset)
        tree_pointer = Node(split_attribute, split_value)
        left_dataset, right_dataset = split_dataset(training_dataset, split_attribute, split_value)
        
        tree_pointer.left, left_depth = decision_tree_learning(left_dataset, depth+1)
        tree_pointer.right, right_depth = decision_tree_learning(right_dataset, depth+1)
        return tree_pointer, max(left_depth, right_depth)
   


def main():
    dataset = np.loadtxt("wifi_db/clean_dataset.txt")
    # Call the decision_tree_learning function to build the decision tree
    tree, depth = decision_tree_learning(dataset, depth=0)

    # You can now traverse the tree and make predictions or explore its structure
    # For example, let's traverse the leftmost branch until a leaf is reached:
    current_node = tree
    while current_node.attribute is not None:
        print(f"Attribute: {current_node.attribute}, Value: {current_node.value}")
        current_node = current_node.left

    # Print the final leaf node
    print(f"Leaf Node Value: {current_node.value}")



if __name__ == "__main__":
    main()