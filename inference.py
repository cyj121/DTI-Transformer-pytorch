import torch
from dataset import get_inference
from model.config_cnn import ConfigCNN
from model.config_drug import ConfigDrug
from model.config_classify import ConfigClassify
from model.model_classify import Classify

inference_loader = get_inference()

config_drug = ConfigDrug()
config_cnn = ConfigCNN()
config_classify = ConfigClassify()

classify_net = Classify(config_drug, config_cnn, config_classify)

classify_net.load_state_dict(torch.load('./trained_model/best_model.pth'))

with torch.no_grad():
    for data in inference_loader:
        drug, protein, labels = data
    output = classify_net(drug, protein)

label_dict = {0: '无相互作用', 1: '有相互作用'}

print(output)