import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as TF
import torch


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
        self.linear = nn.Linear(2048, 300)
        self.softmax = nn.Softmax(dim=1)

        self.resnet = nn.Sequential(*modules)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, images):
        out = self.resnet(images)
        return out
    

class Decoder(nn.module):
    def __init__(self, hidden_size, embedding_size, vocab_size, num_layers, model_type):
        super(Decoder, self)._init_()
        self.embedding  = nn.Embedding(vocab_size, embedding_size)
        self.fc         = nn.Linear(hidden_size, vocab_size)
        if model_type == 'RNN':
            self.recurrent = nn.RNN(input_size = embedding_size, hidden_size = hidden_size, num_layers = num_layers)
        else:
            self.recurrent = nn.LSTM(input_size = embedding_size, hidden_size = hidden_size, num_layers = num_layers)

    def forward(self, features, captions):
        features   = features.unsqueeze(1)
        embeddings = self.embedding(captions)
        embeddings = torch.cat(features, embeddings[:,:-1,:], dim =1)
        hiddens, c = self.recurrent(embeddings)
        return self.fc(hiddens)


class Encoder_Decoder(nn.Module):
  def _init_(self, hidden_size, embedding_size, num_layers, model_type, vocab_size):
    super(Encoder_Decoder, self)._init_()
    self.encoder = Resnet50(embedding_size)
    self.decoder = Decoder(hidden_size, embedding_size, num_layers, model_type)
  def forward(self, images, captions):
    features = self.encoder(images)
    outputs = self.decoder(features, captions)
    return outputs