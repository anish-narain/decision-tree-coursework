# Machine Learning Decision Tree Coursework 1

## Summary 
This repository shows the visualisation and evaluation of a given clean and noisy data set. The dataset presents a value of wifi strength from 7 different wifi signals which can then be used to identify which, of 4 rooms, a person is in. Using decision trees and evalusation metrics we were able to determine the efficiency of our classification and evalusation model in python. We used libaries such as, numPy and matplotlib in order to execute calculations and visulaise the tree.

## Project Setup
To set up our project environment and libaries we used the following commands, having downloaded the requirements.txt file:
```
$ cd your_project_directory 
$ python3 -m venv venv
$ source venv/bin/activate
(venv) $ pip install --upgrade pip
(venv) $ pip install -r requirements.txt
(venv) $ python3 -c "import numpy as np; import torch; print(np); print(torch)"
```
## Individual Functions Explained 

Functions for running the most important instructions are shown below. Please review the individual comments in the source code for more information.

### Decsion Tree and Visualization

Thw code defines two classes, `Node` and `DecisionTree`, to define and build the decsion tree and provides methods for visualizing decision trees. The `Node class` represents a node in a decision tree, storing information about the emitter, value, room, and child nodes. The `DecisionTree class` initializes with a root node and offers two visualization methods. The `visualize_tree` method displays the structure of the decision tree, highlighting leaf nodes and decision rules, and the plot_tree method generates a visual representation of the tree using matplotlib. The class definition is shown below:
```
class Node:
    def __init__(self, emitter, value):
        self.emitter = emitter
        self.value = value
        self.room = None
        self.left = None
        self.right = None
```
Additionally, in order to build the tree the code contains two functions, `reached_leaf` and `decision_tree_learning`, which are essential components of a decision tree learning algorithm. The `reached_leaf` function checks if the current node in the decision tree is a leaf node by comparing the labels of instances in the training dataset. If all labels are the same, it returns True, indicating that a leaf node has been reached.

The `decision_tree_learning` function is used to construct a decision tree recursively. It first checks if the current node is a leaf node using the reached_leaf function. If it is a leaf node, it creates a leaf node with a label. If not, it finds the best emitter and split value for the current node using the find_split function. If the split value is 0, it creates a leaf node with the majority class label. Otherwise, it constructs the left and right child nodes by recursively calling decision_tree_learning and continues to build the decision tree. The function returns the root node of the decision tree and its depth. In summary, this code represents the core logic for building a decision tree based on a training dataset and is a fundamental part of decision tree construction algorithms.

### Split Function
n

### Cross Validation
n

### Evaluation Metrics 
The script defines a series of functions for evaluating the performance of classification models using metrics like a confusion matrix, accuracy, precision, recall, and F1-score. The `confusion_matrix` function calculates the confusion matrix based on the true class labels and predicted class labels, where the row defines the true value and the column is the predicted. The `accuracy_from_confusion` function computes the classification accuracy, while `precision_from_confusion` and `recall_from_confusion` calculate precision and recall values for each class and their macro-averaged values. Finally, the `f1_score_from_confusion` function computes the F1-score for each class and its macro-averaged value. These functions are essential for assessing the quality and effectiveness of classification models, making them a valuable tool for machine learning and classification tasks. Below are the formulae used to compute the metrics described:

![alt text](![image](https://github.com/anish-narain/decision-tree-coursework/assets/115703122/69f8a5d2-0f46-4711-8748-9742f659a12b)
)

## Results 
n
