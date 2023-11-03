from evaluation import *
from decision_tree import *


def plot_tree(dataset, rg):
    """
    Visualises and plots the decision tree using the provided dataset.

    Parameters:
    - dataset: NumPy array, the input dataset.
    - random_generator: RNG object, random number generator for shuffling and splitting.
    """
    testing_dataset, training_dataset = split_training_testing_data(dataset, 0.2, rg)

    # Call the decision_tree_learning function to build the decision tree
    root, depth = decision_tree_learning(training_dataset, depth=0)

    # Visualisation of tree in terminal
    # print(root.visualise_tree())
    root.plot_tree()


def evaluate_and_output(dataset, rg):
    """
    Evaluates the decision tree using k-fold cross-validation and outputs metrics.

    Parameters:
    - dataset: NumPy array, the input dataset for evaluation.
    - random_generator: RNG object, random number generator for cross-validation splits.
    """
    acc, avg_prec, avg_rec, avg_f1, confusion_mat, prec, rec, f1 = cross_validation(dataset, rg)

    print(f"Accuracy: {acc:.2f}\n")
    print("Precision for each room:")
    for i, precision in enumerate(prec):
        print(f"  Room {i + 1}: {precision:.4f}")

    print(f"Average Precision: {avg_prec:.4f}\n")

    print("Recall for each room:")
    for i, recall in enumerate(rec):
        print(f"  Room {i + 1}: {recall:.4f}")

    print(f"Average Recall: {avg_rec:.4f}\n")

    print("F1 Measure for each room:")
    for i, f1_measure in enumerate(f1):
        print(f"  Room {i + 1}: {f1_measure:.4f}")

    print(f"Average F1 Measure: {avg_f1:.4f}\n")

    print("Confusion matrix:")
    print(confusion_mat)


def main():
    seed = 60012
    rg = default_rng(seed)

    dataset = np.loadtxt("wifi_db/clean_dataset.txt")
    print("Visualisation and Evaluation for Clean Dataset:")
    plot_tree(dataset, rg)
    evaluate_and_output(dataset, rg)

    dataset = np.loadtxt("wifi_db/noisy_dataset.txt")
    print("Visualisation and Evaluation for Noisy Dataset:")
    plot_tree(dataset, rg)
    evaluate_and_output(dataset, rg)


if __name__ == "__main__":
    main()
