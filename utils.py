import numpy as np
import matplotlib.pyplot as plt; plt.style.use("fivethirtyeight")
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import accuracy_score, f1_score, auc, roc_auc_score, roc_curve


def get_label(data):
    label = np.zeros_like(data["label"], dtype=np.int32)
    label[data["label"] == "pants-fire"] = 1
    return label


def plot_roc(model, x, y):
    probs = model.predict_proba(x)
    preds = probs[:,1]
    fpr, tpr, threshold = roc_curve(y, preds)
    roc_auc = auc(fpr, tpr)

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    
    
def evaluate(model, x, y):
    y_pred = model.predict(x)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    print("Accuracy: {:.4f}".format(acc))
    print("F-1 score: {:.4f}".format(f1))
    plot_roc(model, x, y)
    
    

def impute(x_array):
    return np.array(["unknown" if x is np.nan else x for x in x_array.values])


imputer = FunctionTransformer(impute)
