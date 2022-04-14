from model.model_encoder import *
from model.model_cnn import *
from torch import nn


class Classify(nn.Module):
    def __init__(self, config_drug, config_cnn, config_classify):
        super(Classify, self).__init__()
        self.config_drug = config_drug
        self.config_cnn = config_cnn
        self.config_classify = config_classify

        self.encoder1 = Encoder1(self.config_drug)
        self.encoder2 = Encoder2(self.config_classify)
        self.cnn = TextCNN(self.config_cnn)

        # Fully-Connected Layer
        self.fc1 = nn.Linear(self.config_cnn.channels_num[2], self.config_classify.fc[0])
        self.fc2 = nn.Linear(self.config_classify.fc[0], self.config_classify.fc[1])
        self.fc3 = nn.Linear(self.config_classify.fc[1], self.config_classify.output_size)

    def forward(self, input_drug, input_protein):
        query = self.encoder1(input_drug)
        query = query.unsqueeze(1)

        value = self.cnn(input_protein)
        value = value.unsqueeze(1)
        key = value

        output = self.encoder2(query, key, value)
        output = output.squeeze()

        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        return output