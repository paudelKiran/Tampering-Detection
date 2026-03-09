import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    matthews_corrcoef
)


class TamperingEvaluation:

    def __init__(self, y_true, y_pred, y_prob=None):
        """
        y_true : ground truth mask (0 or 1)
        y_pred : predicted binary mask (0 or 1)
        y_prob : predicted probability map 
        """

        self.y_true = y_true.flatten()
        self.y_pred = y_pred.flatten()
        self.y_prob = y_prob.flatten() if y_prob is not None else None

        self.cm = confusion_matrix(self.y_true, self.y_pred)

        self.TN, self.FP, self.FN, self.TP = self.cm.ravel()


  
    # Basic Metrics


    def accuracy(self):
        return accuracy_score(self.y_true, self.y_pred)

    def precision(self):
        return precision_score(self.y_true, self.y_pred)

    def recall(self):
        return recall_score(self.y_true, self.y_pred)

    def f1(self):
        return f1_score(self.y_true, self.y_pred)

    def pixel_accuracy(self):
        return np.mean(self.y_true == self.y_pred)



    # IoU (Jaccard)
   

    def iou(self):
        intersection = np.logical_and(self.y_true, self.y_pred)
        union = np.logical_or(self.y_true, self.y_pred)

        if np.sum(union) == 0:
            return 0

        return np.sum(intersection) / np.sum(union)



    # Dice Score
 

    def dice(self):
        intersection = np.sum(self.y_true * self.y_pred)

        return (2 * intersection) / (
            np.sum(self.y_true) + np.sum(self.y_pred) + 1e-8
        )


    # Matthews Correlation


    def mcc(self):
        return matthews_corrcoef(self.y_true, self.y_pred)


    # ROC Curve

    def get_roc_curve(self):

        if self.y_prob is None:
            print("Probability predictions required for ROC")
            return None

        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_prob)

        auc_score = roc_auc_score(self.y_true, self.y_prob)

        return fpr, tpr, auc_score


    
    #  Confusion Matrix
 

    def plot_confusion_matrix(self):

        plt.figure(figsize=(6,5))

        sns.heatmap(
            self.cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Authentic", "Tampered"],
            yticklabels=["Authentic", "Tampered"]
        )

        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")

        plt.show()


    # ROC Curve Plotting

    def plot_roc(self):

        if self.y_prob is None:
            print("Probability predictions required for ROC")
            return

        result = self.get_roc_curve()
        if result is None:
            return
        
        fpr, tpr, auc_score = result

        plt.figure(figsize=(6,5))

        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
        plt.plot([0,1], [0,1], linestyle="--")

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")

        plt.legend()
        plt.show()




    def print_all_metrics(self):

        print("\n===== Evaluation Metrics =====\n")

        print("Confusion Matrix")
        print(self.cm)

        print("\nAccuracy:", self.accuracy())
        print("Precision:", self.precision())
        print("Recall:", self.recall())
        print("F1 Score:", self.f1())

        print("\nIoU:", self.iou())
        print("Dice Score:", self.dice())

        print("\nPixel Accuracy:", self.pixel_accuracy())

        print("\nMatthews Correlation Coefficient:", self.mcc())

        if self.y_prob is not None:
            print("\nAUC:", roc_auc_score(self.y_true, self.y_prob))



