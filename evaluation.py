# evaluation.py
import math
import torch


def count(predicts, labels):
    confusion_matrix = torch.zeros(2, 2)
    for p, l in zip(predicts.view(-1), labels.view(-1)):
        confusion_matrix[p.long(), l.long()] += 1
    tp = confusion_matrix[1][1]
    fp = confusion_matrix[1][0]
    fn = confusion_matrix[0][1]
    tn = confusion_matrix[0][0]
    return tp, fp, fn, tn


def accuracy(tp, fp, fn, tn):
    acc = (tp + tn) / (tp + fp + fn + tn)
    return acc


def precision(tp, fp):
    prec = tp / (tp + fp)
    return prec


def recall(tp, fn):
    re = tp / (tp + fn)
    return re


def specificity(tn, fp):
    spe = tn / (tn + fp)
    return spe


def mcc(tp, fp, fn, tn):
    denominator = tp * tn - fp * fn
    molecule = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    m = denominator / molecule
    return m