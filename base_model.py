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

        # lstm cell
        if model_type == 'LSTM':
            self.lstm_cell = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
        elif model_type == 'RNN':
            self.lstm_cell = nn.RNN(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers)
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
        outputs = outputs.to(device)

        # embed the captions
        captions_embed = self.embed(captions)

        # pass the caption word by word
        for t in range(captions.size(1)):

            # for the first time step the input is the feature vector
            if t == 0:
                hidden_state, cell_state = self.lstm_cell(features)

            # for the 2nd+ time step, using teacher forcer
            else:
                hidden_state, cell_state = self.lstm_cell(captions_embed[:, t, :], cell_state)

            # output of the attention mechanism
            out = self.fc_out(hidden_state)

            # build the output tensor
            outputs[:, t, :] = out

        return outputs

    def predict(self, features, max_length=20, deterministic=False, temperature=1.0):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # batch size
        batch_size = features.size(0)
        final_output = []

        t = 0
        while True:
            if t == 0:
                hidden_state, cell_state = self.lstm_cell(features)
                t = t + 1
            else:
                hidden_state, cell_state = self.lstm_cell(features, cell_state)
            out = self.fc_out(hidden_state)
            out.squeeze_(1)
            if deterministic:  # deterministically sample from softmax
                # returns values, indices, we only want indices to decode using vocab
                _, max_idx = torch.max(out, dim=1)  # 1d array size N
            else:  # use temperature in softmax and sample
                # calc softmax w/ temperature
                softmax = torch.softmax(out / temperature, dim=1)  # (N, vocab_size)
                # sample softmax
                max_idx = Categorical(softmax).sample()  # 1d array size N
            final_output.append(max_idx)
            if len(final_output) >= max_length:
                break

            features = self.embed(max_idx)
        return torch.stack(final_output).permute(1, 0)


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
