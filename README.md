# Machine Learning Decision Tree Coursework 1

## Summary 
This repository shows the visualisation and evaluation of a given clean and noisy data set. The dataset presents a value of wifi strength from 7 different wifi signals which can then be used to identify which, of 4 rooms, a person is in. Using decision trees and evaluation metrics we were able to determine the efficiency of our classification and evaluation model in python. We used libraries such as, NumPy and Matplotlib in order to execute calculations and visualise the tree.

## Running the Code
After extracting the zip file, navigate to the extracted folder in the directory.

1. To view the visualisation of the decision tree from Matplotlib and a printed version in the terminal, ensure `Tests.test_tree_plot(dataset, rg)` is uncommented from `main()` in `main.py`.
2. To view the Cross Validation classification metrics, ensure `Tests.test_cross_validation(dataset, rg)` is uncommented from `main()` in `main.py`.

Modify the file name in `main()` to run the code on the required file `dataset = np.loadtxt("ENTER FILE NAME")`

To run the code:

```
python3 main.py
```
## Individual Functions Explained 

Functions for running the most important instructions are shown below. Please review the individual comments in the source code for more information.

### Decision Tree and Visualization

The code defines on class, `TreeNode`, to define and build the decision tree and provides methods for visualizing decision trees. The class represents a node in a decision tree, storing information about the emitter, value, room, and child nodes. It also offers two visualization methods. The `visualize_tree` method displays the structure of the decision tree, highlighting leaf nodes and decision rules, and the plot_tree method generates a visual representation of the tree using matplotlib. The class definition is shown below:
```
class TreeNode:
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
The `find_split` function identifies the optimal split in a dataset. It iterates through features, evaluating potential split points using information gain. The function returns the best feature index and corresponding split point, crucial for accurate decision tree construction.

### Cross Validation
The `cross_validation` function performs 10-fold cross-validation. It splits the dataset into training and test sets, trains a decision tree on the training data, and evaluates its performance using metrics like accuracy, precision, recall, and F1-score. The function returns the average values of these metrics over the 10 folds, providing a robust assessment of the model's effectiveness.

### Evaluation Metrics 
The script defines a series of functions for evaluating the performance of classification models using a confusion matrix, accuracy, precision, recall, and F1-score. The `confusion_matrix` function calculates the confusion matrix based on the true class labels and predicted class labels, where the row defines the true value and the column is the predicted. The `accuracy_from_confusion` function computes the classification accuracy, while `precision_from_confusion` and `recall_from_confusion` calculate precision and recall values for each class and their macro-averaged values. Finally, the `f1_score_from_confusion` function computes the F1-score for each class and its macro-averaged value. These functions are essential for assessing the quality and effectiveness of classification models, making them a valuable tool for machine learning and classification tasks. Below are the formulae used to compute the metrics described:

![Alt Text](https://raw.githubusercontent.com/KennyMiyasato/classification_report_precision_recall_f1-score_blog_post/b059ac3f2ac16780d4deb2405060513b7cd2813c/images/accuracy_formula.svg)

![Alt Text](https://raw.githubusercontent.com/KennyMiyasato/classification_report_precision_recall_f1-score_blog_post/b059ac3f2ac16780d4deb2405060513b7cd2813c/images/precision_formula.svg)

![Alt Text](https://github.com/KennyMiyasato/classification_report_precision_recall_f1-score_blog_post/raw/master/images/recall_formula.svg)

![Alt Text](https://github.com/KennyMiyasato/classification_report_precision_recall_f1-score_blog_post/raw/master/images/f1-score_formula.svg)

