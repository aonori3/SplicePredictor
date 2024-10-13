import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from itertools import cycle
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    roc_auc_score, roc_curve, average_precision_score
)
import tensorflow as tf

def plot_performance_graphs(n_epochs, history):
    """Plot training and validation accuracy and loss."""
    plt.style.use("ggplot")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Accuracy plot
    ax1.plot(range(n_epochs), history.history["accuracy"], label="train_acc")
    ax1.plot(range(n_epochs), history.history["val_accuracy"], label="val_acc")
    ax1.set_title("Training and Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    
    # Loss plot
    ax2.plot(range(n_epochs), history.history["loss"], label="train_loss")
    ax2.plot(range(n_epochs), history.history["val_loss"], label="val_loss")
    ax2.set_title("Training and Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

def calculate_metrics(model, x_test, y_test, normalize=False, graph=True):
    """Calculate and display various metrics."""
    labels = ['donor', 'acceptor', 'other']
    prob_predictions = model.predict(x_test)
    class_preds = tf.greater(prob_predictions, 0.5)
    
    print('Classification Report:')
    print(classification_report(y_test, class_preds))
    
    cm = confusion_matrix(y_test.argmax(axis=1), prob_predictions.argmax(axis=1), 
                          normalize='true' if normalize else None)
    pd_cm = pd.DataFrame(cm, index=labels, columns=labels)

    if graph:
        plot_confusion_matrix(pd_cm, normalize)

    print("---Calculating other metrics---")
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)

    metrics = {
        "Specificity": tn / (tn + fp),
        "Sensitivity": tp / (tp + fn),
        "Precision": tp / (tp + fp),
        "F1": (2 * tp) / ((2 * tp) + fp + fn)
    }

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

def plot_confusion_matrix(pd_cm, normalize):
    """Plot confusion matrix."""
    print('\nConfusion Matrix:')
    plt.figure(figsize=(15, 10))
    format = ".4f" if normalize else "d"
    g = sns.heatmap(pd_cm, annot=True, cmap="Blues", fmt=format, annot_kws={"fontsize": 15})
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize=13)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize=13)
    plt.yticks(rotation=0)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

def plot_curve_for_all_classes(x, y, x_label, y_label, general_area, plot_title, n_classes=3):
    """Plot ROC or PR curve for all classes."""
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

    plt.figure(figsize=(7, 8))
    lines, labels = [], []
    
    # Plot micro-average
    l, = plt.plot(x['micro'], y['micro'], color='gold', lw=2)
    lines.append(l)
    labels.append(f"{plot_title} (area = {general_area['micro']:.2f})")

    # Plot each class
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(x[i], y[i], color=color, lw=2)
        lines.append(l)
        labels.append(f"{plot_title} for class {i} (area = {general_area[i]:.2f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{plot_title} curve")
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    plt.tight_layout()
    plt.show()

def calculate_roc_auc_pr(model, x_test, y_test, plot_roc=True, plot_pr=True):
    """Calculate ROC AUC and PR AUC metrics."""
    y_pred = model.predict(x_test)
    n_classes = y_pred.shape[1]
    
    precision, recall, average_precision = {}, {}, {}
    fpr, tpr, roc_auc = {}, {}, {}

    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:,i], y_pred[:,i])
        average_precision[i] = average_precision_score(y_test[:,i], y_pred[:,i])
        fpr[i], tpr[i], _ = roc_curve(y_test[:,i], y_pred[:,i])
        roc_auc[i] = roc_auc_score(y_test[:,i], y_pred[:,i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = roc_auc_score(y_test, y_pred, average='micro')

    precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_pred.ravel())
    average_precision["micro"] = average_precision_score(y_test, y_pred, average='micro')

    if plot_roc:
        plot_curve_for_all_classes(fpr, tpr, "False Positive Rate", "True Positive Rate", roc_auc, "ROC-AUC", n_classes)
    
    if plot_pr:
        plot_curve_for_all_classes(recall, precision, "Recall", "Precision", average_precision, "Precision-Recall", n_classes)

    print('ROC AUC: ', roc_auc)
    print('PR AUC: ', average_precision)

    return average_precision, roc_auc