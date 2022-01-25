import glob
import sys
import os
import numpy as np
from sklearn import svm
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import pandas as pd

def plot_confusion_matrix(test_y, pred_y, class_names, normalize=False):
    cm = confusion_matrix(test_y, pred_y)
    classes = class_names[unique_labels(test_y,pred_y)]
    classes = [ flatten for inner in classes for flatten in inner ]
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label\n',
           xlabel='\nPredicted label')
    fmt = '.2f' if normalize else 'd'
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center")
    fig.tight_layout()
    return ax

if __name__ == "__main__":
    help_desc_msg ="""== SVMを利用するためのプログラム == 
       svm_sample.py (本ファイル)
       train.csv　学習の特徴量データ
       train_label.csv　学習のラベルデータ
       valid.csv　テストデータ
       valid_label.csv テストデータのラベル
       """
	
    help_epi_msg = """説明完了"""

    parser = argparse.ArgumentParser(
    	description=help_desc_msg,
    	epilog=help_epi_msg,
    	formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t','--train', default='train.csv', help='学習データ。デフォルト：train.csv')
    parser.add_argument('-v','--valid', default='valid.csv', help='検証データ。デフォルト：valid.csv')
    parser.add_argument('-tl','--trainlabel', default='train_label.csv', help='学習ラベル。デフォルト：train.csv')
    parser.add_argument('-vl','--validlabel', default='valid_label.csv', help='検証ラベル。デフォルト：valid.csv')
    parser.add_argument('-n','--name', default='name.txt', help='クラス名。デフォルト：name.txt')
    parser.add_argument('-o','--out', default='result', help='出力ファイル名の指定。'\
    					'拡張子は指定しない。'\
    					'デフォルトはresult')
    parser.add_argument('-m','--mode', default=0, type=int, help='デフォルト(0): 線形SVM、1：非線形（RBF）')
    parser.add_argument('-sep','--sep', default=0, type=int, help='CSV形式 デフォルト(0): , 1：\t')
    parser.add_argument('-c','--C', default=1, type=float, help='デフォルト(1)')
    parser.add_argument('-g','--gamma', default=0.001, type=float, help='デフォルト(0.001)')
    
    args = parser.parse_args()
    
    deli = ','
    if args.sep == 1:
        deli = '\t'
    
    ker = 'linear'
    if args.mode == 1:
        ker = 'rbf'
    
    
    #csvファイルの読み込み
    train = np.loadtxt(args.train, delimiter = deli, dtype = "float")
    valid = np.loadtxt(args.valid, delimiter = deli, dtype = "float")
    train_label = np.loadtxt(args.trainlabel, delimiter = deli, dtype = "int")
    valid_label = np.loadtxt(args.validlabel, delimiter = deli, dtype = "int")
    #class_names = np.loadtxt(args.name, delimiter = deli, dtype = "str",encoding="UTF-8")
    class_names = pd.read_csv(args.name, sep = deli, index_col=False, header=None).values
    #print(class_names)
    #SVM
    model = svm.SVC(kernel = ker, C=args.C, gamma = args.gamma)
    model.fit(train, train_label)
    
    #計算する
    ans = model.predict(valid)
    
    #結果について
    print("Accuracy = ", end='')
    print(accuracy_score(valid_label, ans))
    print("Precision = ", end='')
    print(precision_score(valid_label, ans))
    print("Recall = ", end='')
    print(recall_score(valid_label, ans))
    print("F score = ", end='')
    print(f1_score(valid_label, ans))
    plot_confusion_matrix(valid_label, ans, class_names)
    plt.show()
    
    #print(ans)
    