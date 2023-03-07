import torchvision.models as models
import torch.nn as nn
import torch

from vocab import Vocabulary


# The baseline model
# Resnet50 for encoding
# LSTM for decoding
# Writeup recommends:
# 2 layers
# 512 hidden units
class BaselineModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, vocab: Vocabulary, num_layers=2):
        super(BaselineModel, self).__init__()
        self.num_layers = num_layers
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.hidden_size = hidden_size
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters(): # freeze params
            param.requires_grad = False
        resnet.fc = nn.Linear(resnet.fc.in_features, num_layers * hidden_size)
        self.resnet = resnet
        self.word_embedder = nn.Embedding(self.vocab_size, embedding_size)
        self.decoder = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # if batch first set to false it will accept
        # and output tensors w/ size (L, N, I) where
        # L is size of sequence
        # N is batch size
        # I is size of each vector in sequence
        # if batch_first=True then tensors will be
        # of the form (N, L, I), which is just nicer
        # to deal w/
        # for feeding hiddens into
        self.hidden2output = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, images, captions):
        """
        Forward pass. nuff' said.
        Params:
            images: 2d tensor w/ shape (N, img_size)
            captions: 32 tensor w/ shape (N, seq len)
                A batch of target captions for the corresponding
                image. Each sequence should be a list of integer indices
                obtained by running through tokenized caption
                through vocabulary object. padding where appropriate
        Outputs:
            3d tensor w/ shape (N, vocab size, seq len)
                A tensor containing the one-hot encoded word
                predictions. This specific ordering used so
                that it can be fed into the loss function.
        """
        # encode images to LSTM hidden state
        batch_size = images.size()[0]
        hidden_states = self.resnet(images) # (N, num_layers * hidden_size)
        hidden_states = torch.reshape(hidden_states, (batch_size, self.num_layers, self.hidden_size))
        # for some fucking reason turning on batch_first lets me feed in
        # batch first inputs but not hiddens? mfw stupid as fuck moment
        hidden_states = torch.permute(hidden_states, (1, 0, 2)).contiguous()
        cell_states = hidden_states.clone().contiguous() # ¯\_(:P)_/¯
        embedded = self.word_embedder(captions) # (N, seq len, embedding_size)
        # print(f'embedded: {embedded.size()}')
        # decode w/ lstm
        # (N, seq len, hidden_size)
        final_hiddens, (_, _) = self.decoder(embedded, (hidden_states, cell_states))
        # print(f'final_hiddens: {final_hiddens.size()}')
        output = self.hidden2output(final_hiddens) # (N, seq len, vocab size)
        return output.permute(0, 2, 1)

    def generate(self, images, config):
        """
        Generate captions for the images in the batch
        based completely off hidden states encoded by
        the model.
        Params:
            images: 2d tensor w/ shape (N, img_size)
            config: The generation config. See "generation"
                in default.json for more details
        Output:

        """
        # read config stuff.
        max_len = config['max_length']
        deterministic = config['deterministic']
        temperature = config['temperature']

        # encode images to LSTM hidden state
        batch_size = images.size()[0]
        hidden_states = self.resnet(images)  # (N, num_layers * hidden_size)
        hidden_states = torch.reshape(hidden_states, (batch_size, self.num_layers, self.hidden_size))
        cell_states = hidden_states.clone()  # ¯\_(:P)_/¯

        # convert <start> into a word and we can begin
        # TODO
        #  feed whatever index associated w/ <start> in vocab,
        #  then cache latest hidden and cell state and feed the
        #  next predicted word embedding as input.
        pass
