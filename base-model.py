import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as TF


class Resnet50(nn.Module):
    def __init__(self, n_class, freeze_encoder=True):
        self.n_class = n_class
        self.freeze_encoder = freeze_encoder
        super(Resnet50, self).__init__()
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')

        if self.freeze_encoder:
            for param in resnet.parameters():
                param.requires_grad = False
        modules = list(resnet.children())[:-1]
        self.linear1 = nn.Linear(2048, 300)
        self.lstm = nn.LSTM(input_size=300, hidden_size=512, num_layers=2, batch_first=True)
        self.linear2 = (512, n_class)
        self.softmax = nn.Softmax(dim=1)


        self.resnet = nn.Sequential(*modules)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, images):
        out = self.resnet(images)
        return out
