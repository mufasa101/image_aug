import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def compare_validation_accuracy(original_history, augmented_history):
    print("📊 Comparing validation accuracy for both models...")

    # creating a side-by-side graph of accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(original_history.history['val_accuracy'], label='Original Dataset', color='blue')
    plt.plot(augmented_history.history['val_accuracy'], label='Augmented Dataset', color='green')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def compare_validation_loss(original_history, augmented_history):
    print("📉 Comparing validation loss for both models...")
    # creating a side-by-side graph of loss
    plt.figure(figsize=(10, 5))
    plt.plot(original_history.history['val_loss'], label='Original Dataset', color='blue')
    plt.plot(augmented_history.history['val_loss'], label='Augmented Dataset', color='green')
    plt.title('Validation Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Show confusion matrix for the augmented model (to see where it did well or messed up)
def show_confusion_matrix(test_labels_encoded, predictions, label_encoder):
    print("📉 Making confusion matrix...")
    cm = confusion_matrix(test_labels_encoded, predictions)  # Compare true vs predicted
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title("Confusion Matrix - Augmented Model")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.show()

# Example usage:
# 1. Pass in the histories of the two models to compare_accuracy() and compare_loss()
# 2. Pass in the test labels, predictions, and label encoder to show_confusion_matrix()
