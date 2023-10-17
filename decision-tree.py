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

(x, y, classes) = read_dataset("./wifi_db/clean_dataset.txt")
print(x, "\n")
print(y, "\n")
print(classes, "\n")

class Node:
    def __init__(self, attribute, value):
        self.attribute = attribute
        self.value = value
        self.left = None
        self.right = None

class DecisionTree:
    def __init__(self, root):
        self.root = Node(root)
    
    def visualize_tree(self, node, level=0, prefix="Root: "):
        if node is not None:
            print(" " * (level * 4), prefix, "Wifi Signal: ", str(node.attribute), " < ", str(node.value))
            if node.left is not None or node.right is not None:
                self.visualize_tree(node.left, level + 1, "L--- ")
                self.visualize_tree(node.right, level + 1, "R--- ")

def find_split(training_dataset):
    return (4, -0.64) #random input

def decision_tree_learning(training_dataset, depth):
    split_attribute, split_value = find_split(training_dataset)
    

def main():
    # Create a simple decision tree
    tree = DecisionTree("Outlook")
    tree.root.left = Node("Temperature", "80")
    tree.root.right = Node("Humidity", "70")
    tree.root.left.left = Node("Windy", "True")
    tree.root.left.right = Node("Play Golf", "No")
    tree.root.right.left = Node("Windy", "False")
    tree.root.right.right = Node("Play Golf", "Yes")

    # Visualize the tree
    tree.visualize_tree(tree.root)

if __name__ == "__main__":
    main()