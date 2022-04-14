import torch
from tqdm import tqdm
from dataset import get_test
from model.config_cnn import ConfigCNN
from model.config_drug import ConfigDrug
from model.config_classify import ConfigClassify
from model.model_classify import Classify
from evaluation import *


test_loader = get_test()

config_drug = ConfigDrug()
config_cnn = ConfigCNN()
config_classify = ConfigClassify()

classify_net = Classify(config_drug, config_cnn, config_classify)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

classify_net.to(device)

classify_net.load_state_dict(torch.load('./trained_model/best_model.pth'))

tn_sum = 0
fn_sum = 0
tp_sum = 0
fp_sum = 0

with torch.no_grad():
    classify_net.eval()
    for batch in tqdm(test_loader):
        drugs, proteins, labels = batch

        drugs, proteins, labels = drugs.to(device), proteins.to(device), labels.to(device)

        outputs = classify_net(drugs, proteins)
        predicts = torch.argmax(outputs, dim=1)
        tp, fp, fn, tn = count(predicts, labels)
        tp_sum += tp
        fp_sum += fp
        fn_sum += fn
        tn_sum += tn

prec = precision(tp_sum, fp_sum)
acc = accuracy(tp_sum, fp_sum, fn_sum, tn_sum)
re = recall(tp_sum, fn_sum)
spe = specificity(tn_sum, fp_sum)
mc = mcc(tp_sum, fp_sum, fn_sum, tn_sum)
print('precision：', prec)
print('accuracy：', acc)
print('recall：', re)
print('specificity：', spe)
print('mcc：', mc)