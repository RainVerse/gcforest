from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, auc
import matplotlib.pyplot as plt


def plot_roc_curve(y_test, y_proba):
    fig2 = plt.figure(figsize=(12, 6))
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_proba)
    roc_auc = auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    fig2.show()
    return roc_auc


def plot_pr_curve(y_test, y_proba):
    precision, recall, _ = precision_recall_curve(y_true=y_test, probas_pred=y_proba)
    average_precision = average_precision_score(y_test, y_proba)
    plt.figure(figsize=(12, 6))
    plt.step(recall, precision, color='r', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='#F59B00')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('OverSampling Precision-Recall curve: \n Average Precision-Recall Score ={0:0.2f}'.format(
        average_precision), fontsize=16)
    plt.show()
    return average_precision
