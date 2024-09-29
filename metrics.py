import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, auc, roc_curve, average_precision_score

def performance_graphs(n_epochs, H):
    plt.style.use("ggplot")
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(0, n_epochs), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, n_epochs), H.history["val_accuracy"], label="val_acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(0, n_epochs), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, n_epochs), H.history["val_loss"], label="val_loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def other_metrics(model, x_test, y_test, normalize=False, graph=True):
    labels = ['donor', 'acceptor', 'other']
    prob_predictions = model.predict(x_test)
    class_preds = tf.greater(prob_predictions, 0.5)
    print('Classification Report: ')
    print(classification_report(y_test, class_preds))
    cm = confusion_matrix(y_test.argmax(axis=1), prob_predictions.argmax(axis=1), normalize='true' if normalize else None)
    pd_cm = pd.DataFrame(cm, index=labels, columns=labels)
    if graph:
        plt.figure(figsize=(15, 10))
        format = ".4f" if normalize else "d"
        g = sn.heatmap(pd_cm, annot=True, cmap="Blues", fmt=format, annot_kws={"fontsize": 15})
        g.set_xticklabels(g.get_xmajorticklabels(), fontsize=13)
        g.set_yticklabels(g.get_ymajorticklabels(), fontsize=13)
        plt.yticks(rotation=0)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.show()
    print("---Calculating other metrics ---")
    fp = cm.sum(axis=0) - np.diag(cm)
    fn = cm.sum(axis=1) - np.diag(cm)
    tp = np.diag(cm)
    tn = cm.sum() - (fp + fn + tp)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = (2 * tp) / ((2 * tp) + fp + fn)
    print("Specificity: ", specificity)
    print("Sensitivity: ", sensitivity)
    print("Precision: ", precision)
    print("F1: ", f1)