import numpy as np
import pandas as pd
import kNN
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


unique_labels = [0,1,2,3,4,5,6,7,8,9]

def data_labels(dataset):
    return (dataset[dataset.columns[1:]].values, dataset["label"].values.reshape(-1,1))

def save_res(cols, rows, name):
    res = pd.DataFrame(rows, columns=cols)
    res.to_csv("res/{}.csv".format(name), index=False, header=True)

# Returns k splits of dataset (indexes)
def Kfold_gen(dataset, K):
    return KFold(n_splits=K, shuffle=False, random_state=None).split(dataset)

def Kfold_split(dataset, split):
    train, test = split
    return (data_labels(dataset.iloc[train]), data_labels(dataset.iloc[test]))

# Run kNN on train data and validate over a list of values for k
def kNN_predict(train_val, ks):
    train_data, train_labels = train_val[0]
    val_data, val_labels = train_val[1]

    # Init classifier
    clf = kNN.KNNClassifier(1,10)
    clf.fit(train_data, train_labels)
    
    acc = []
    precision = []
    recall = []
    f1 = []
    for k in ks:
        # Predict
        clf.setneighbors(k)
        pred_labels = clf.predict(val_data)
        
        # Metrics
        acc.append(accuracy_score(y_true=val_labels, y_pred=pred_labels))
        precision.append(precision_score(y_true=val_labels, y_pred=pred_labels, labels=unique_labels, average='weighted', zero_division=0))
        recall.append(recall_score(y_true=val_labels, y_pred=pred_labels, labels=unique_labels, average='weighted', zero_division=0))
        f1.append(f1_score(y_true=val_labels, y_pred=pred_labels, labels=unique_labels, average='weighted', zero_division=0))
    return np.stack([acc, precision, recall, f1], axis=-1)