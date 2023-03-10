# Usage

Using Python 3.9:

```
python -m venv venv
pip install -r requirements.txt
```

To run the model, run:

```
python main.py [<experiment>]
```

In the project's working directory. The `<experiment>` argument 
defines the configuration file to use for the experiment. If none is
provided it will default to `default` which uses the `default.json` file.

We have 3 model types: `lstm2`, `rnn`, and `arch2`.

* Define the configuration for your experiment. See `default.json` to see the structure and available options. You are free to modify and restructure the configuration as per your needs.
* Implement factories to return project specific models, datasets based on config. Add more flags as per requirement in the config.
* Implement `experiment.py` based on the project requirements.
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir. This can be configured in `contants.py`
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training pr evaluate performance.


# Image Captioning

Image captioning with LSTMs and RNNs. This repository contains 3 recurrent neural networks for 
image captioning trained on a truncated version of Microsoft's Common Objects in COntext (COCO) 
dataset. We present 3 networks, the general architecture of each being a pretrained Resnet50 network
provided by PyTorch to encode images into an embedding dimension. Then, these image embeddings are
fed into a decoding network and captions are generated word-by-word. Subsequent output words are fed
back into the decoder as inputs during generation, with teacher forcing used instead during training.

1. The `lstm` network uses an LSTM as the decoder.
2. The `rnn` network uses an RNN as the decoder.
3. The `arch2` network uses an LSTM as the decoder, with the image embeddings concatenated onto the word inputs at every time step rather than just the initial time step.