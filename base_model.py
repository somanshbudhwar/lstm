import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as TF
import torch
from torch.distributions import Categorical


class Encoder(nn.Module):
    def __init__(self, embed_size, freeze_encoder=True):
        self.embed_size = embed_size
        self.freeze_encoder = freeze_encoder
        super(Encoder, self).__init__()
        resnet = models.resnet50(weights='ResNet50_Weights.DEFAULT')

        if self.freeze_encoder:
            for param in resnet.parameters():
                param.requires_grad = False
        modules = list(resnet.children())[:-1]

        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(2048, embed_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, images):
        out = self.resnet(images)
        out = out.reshape(out.size(0), -1)
        linear_out = self.relu(self.linear(out))
        return linear_out


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, model_type='LSTM'):
        super(DecoderRNN, self).__init__()

        # define the properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.__temperature =0.1
        self.__sampling = 'stochastic'

        self.num_layers = num_layers

        # lstm cell
        self.lstm_cell = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # output fully connected layer
        self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)

        # embedding layer
        self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)

        # activations

    def forward(self, features, captions):

        # batch size
        batch_size = features.size(0)

        # init the hidden and cell states to zeros
        # hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda()
        # cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()

        # define the output tensor placeholder
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        outputs = torch.empty((batch_size, captions.size(1), self.vocab_size))

        captions_embed = self.embed(captions[:, :-1])
        cell_state = (torch.zeros((2, batch_size, self.hidden_size)).detach(),
                      torch.zeros((2, batch_size, self.hidden_size)).detach())

        inputs = torch.cat((features.unsqueeze(1), captions_embed), dim=1)
        outputs, cell_state = self.lstm_cell(inputs, cell_state)


        outputs = self.fc_out(outputs)

        return outputs

    def predict(self, features, max_length=20):
        # final_output = []
        i=0
        # batch size
        batch_size = features.size(0)
        predicted_captions = torch.zeros((batch_size, max_length))

        cell_state = (torch.zeros((2, batch_size, self.hidden_size)).detach(),
                      torch.zeros((2, batch_size, self.hidden_size)).detach())
        while True:
            hidden_state, cell_state = self.lstm_cell(features.unsqueeze(1), cell_state)
            out = self.fc_out(hidden_state)
            out = out.squeeze_(1)
            if self.__sampling == 'deterministic':
                _, max_idx = torch.max(out, dim=1)
            else:
                softmax = torch.nn.functional.softmax(out / self.__temperature, dim=1)
                max_idx = Categorical(softmax).sample()
            predicted_captions[:, i] = max_idx
            # final_output.extend([max_idx.cpu().numpy()])
            features = self.embed(max_idx)
            # features = features.unsqueeze(1)
            i+=1
            if i >= max_length:
                break
        return predicted_captions


class Encoder_Decoder(nn.Module):
    def __init__(self, hidden_size, embedding_size, num_layers, model_type, vocab_size):
        super(Encoder_Decoder, self).__init__()
        self.encoder = Encoder(embedding_size)
        self.decoder = DecoderRNN(embedding_size, hidden_size, vocab_size, num_layers, model_type)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def predict(self, images):
        features = self.encoder(images)
        outputs = self.decoder.predict(features)
        return outputs
