from base_model import BaselineEncoderDecoder
# Build and return the model here based on the configuration.
from lstm_model import LSTMModel


def get_model(config_data, vocab):
    hidden_size = config_data['model']['hidden_size']
    embedding_size = config_data['model']['embedding_size']
    model_type = config_data['model']['model_type']
    num_layers = config_data['model']['num_layers']

    # You may add more parameters if you want
    # model = BaselineEncoderDecoder(hidden_size, embedding_size, vocab, num_layers, model_type)
    if model_type == "LSTM":
        model = BaselineEncoderDecoder(hidden_size, embedding_size, vocab, num_layers, model_type)
    elif model_type == "LSTM2":
        model = LSTMModel(embedding_size, hidden_size, vocab, num_layers)
    else:
        model = LSTMModel(embedding_size, hidden_size, vocab, num_layers)
    return model
