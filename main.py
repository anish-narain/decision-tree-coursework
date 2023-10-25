from evaluation import *
from decision_tree import *

class Tests:
    def test_tree_plot(dataset, rg):
        testing_dataset, training_dataset = split_training_testing_data(dataset, 0.2, rg)

        # Call the decision_tree_learning function to build the decision tree
        tree, depth = decision_tree_learning(training_dataset, depth=0)

        current_node = tree
    
        new_tree = DecisionTree(current_node.emitter,current_node.value)
        print(new_tree.visualize_tree(current_node))
        new_tree.plot_tree(current_node)
        # Print the final leaf node
        print(f"Leaf Node value: {current_node.value}")
        
    def test_confusion_matrix(dataset):
        true_dataset = []
        prediction_dataset = []

        true_dataset1 = [1.0, 3.0, 2.0, 1.0, 4.0]
        prediction_dataset1 = [3.0, 2.0, 3.0, 1.0, 3.0]

        confusionmatrix = confusion_matrix(true_dataset,prediction_dataset, 4)
        print(confusionmatrix)
        print(accuracy_from_confusion(confusionmatrix))
        print(precision_from_confusion(confusionmatrix))
        print(recall_from_confusion(confusionmatrix))
        print(f1_score_from_confusion(confusionmatrix))

    def test_cross_validation(dataset, rg):
        print("hello new test")
        print(cross_validation(dataset, rg))



def main():
    dataset = np.loadtxt("wifi_db/clean_dataset.txt")
    seed = 60012
    rg = default_rng(seed)

    
    # Tests.test_tree_plot(dataset, rg)
    # Tests.test_confusion_matrix(dataset)
    Tests.test_cross_validation(dataset, rg)
    
    

if __name__ == "__main__":
    main()