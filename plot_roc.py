import numpy as np
import csv
import sys
import json
import argparse
from sklearn.metrics import roc_curve, auc

absolute_path = "result/"
path_if = 'isolateforest_0.01_PortScan_0.7310.json'
path_mlp = 'mlp_0.01_PortScan_0.9156.json'
path_main = 'CUE_SSAD_0.01_PortScan_0.9512_18.json'
path_ssad = 'portscan/results.json'

def load_baseline(path):
    f = open(path, "r")
    p = json.load(f)
    return p

def load_mymethod(path):
    f = open(path, "r")
    p = json.load(f)
    p['dicts_cls']['tag'] = 'CUE-SSAD(CLS)'
    p['dicts_rec']['tag'] = 'CUE-SSAD(REC)'
    p['dicts_combine']['tag'] = 'CUE-SSAD(REC+CLS)'
    p['dicts_conf']['tag'] = 'CUE-SSAD'
    return p['dicts_cls'], p['dicts_rec'], p['dicts_combine'], p['dicts_conf'], p['test_label']

def load_ssad(path):
    f = open(path, "r")
    p = json.load(f)
    l = np.array(p['test_scores'])
    p['test_label'] = list(l[:,1])
    p['test_scores'] = list(l[:,2])
    p['tag'] = 'Deep SSAD'
    return p

def cal_roc(label, predict_score):
    fpr, tpr, _ = roc_curve(label, predict_score)
    roc_auc = auc(fpr,tpr)
       
    print(roc_auc)
    return fpr, tpr, roc_auc
       #savefig([[fpr,tpr,roc_auc]])

def savefig(data, test_label):
    import matplotlib
    #matplotlib.use('Agg')
    matplotlib.rcParams['backend'] = 'SVG'
    import matplotlib.pyplot as plt
    plt.clf()
    colors = ["cornflowerblue","lightslategrey","crimson","rebeccapurple","teal","olive","maroon"]#,"chocolate","darkseagreen"]
    for color, d in zip(colors, data):
        if 'test_label' in d:
            ll = d['test_label']
        else:
            ll =test_label
        fpr,tpr, roc_auc = cal_roc(ll, d['test_scores'])
        plt.plot(fpr, tpr, color = color, label=(d['tag']+"(area=%0.4f)") % roc_auc)
    plt.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig('roc_PortScan'+'.pdf')


def main():
    score_mlp = load_baseline(absolute_path+path_mlp); score_mlp['tag']='Mlp'
    score_if = load_baseline(absolute_path+path_if); score_if['tag']='Isolate Forest'
    score_cls, score_rec, score_combine, score_conf,test_label = load_mymethod(absolute_path+path_main)
    score_ssad = load_ssad(absolute_path+path_ssad)
    data = [score_mlp, score_if, score_ssad, score_rec, score_cls, score_combine, score_conf]
    savefig(data, test_label)


if __name__ == '__main__':
    # main(parse_arguments(sys.argv[1:]))
    main()
