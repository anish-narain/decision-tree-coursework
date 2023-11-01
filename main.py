from evaluation import *
from decision_tree import *


class Tests:
    def test_tree_plot(dataset, rg):
        testing_dataset, training_dataset = split_training_testing_data(dataset, 0.2, rg)

        # Call the decision_tree_learning function to build the decision tree
        root, depth = decision_tree_learning(training_dataset, depth=0)

        print(root.visualize_tree())
        root.plot_tree()
        

    def test_confusion_matrix(dataset):
        true_dataset = []
        prediction_dataset = []

        true_dataset1 = [1.0, 3.0, 2.0, 1.0, 4.0]
        prediction_dataset1 = [3.0, 2.0, 3.0, 1.0, 3.0]

        confusionmatrix = confusion_matrix(true_dataset, prediction_dataset, 4)
        print(confusionmatrix)
        print(accuracy_from_confusion(confusionmatrix))
        print(precision_from_confusion(confusionmatrix))
        print(recall_from_confusion(confusionmatrix))
        print(f1_score_from_confusion(confusionmatrix))

    def test_cross_validation(dataset, rg):
        acc, avg_prec, avg_rec, avg_f1, confusion_mat, prec, rec, f1 = cross_validation(dataset, rg)
        

        print(f"Accuracy: {acc:.2f}\n")
        print("Precision for each room:")
        for i, precision in enumerate(prec):
            print(f"  Room {i+1}: {precision:.4f}")

        print(f"Average Precision: {avg_prec:.4f}\n")

        print("Recall for each room:")
        for i, recall in enumerate(rec):
            print(f"  Room {i+1}: {recall:.4f}")

        print(f"Average Recall: {avg_rec:.4f}\n")

        print("F1 Measure for each room:")
        for i, f1_measure in enumerate(f1):
            print(f"  Room {i+1}: {f1_measure:.4f}")

        print(f"Average F1 Measure: {avg_f1:.4f}\n")

        print("Confusion matrix:")
        print(confusion_mat)
        


def main():
    dataset = np.loadtxt("wifi_db/clean_dataset.txt")
    seed = 60012
    rg = default_rng(seed)

    Tests.test_tree_plot(dataset, rg)
    # Tests.test_confusion_matrix(dataset)
    Tests.test_cross_validation(dataset, rg)


if __name__ == "__main__":
    main()
