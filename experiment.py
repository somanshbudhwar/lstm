import matplotlib.pyplot as plt
import nltk.tokenize
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime

import caption_utils
from caption_utils import *
from constants import ROOT_STATS_DIR
from dataset_factory import get_datasets
from file_utils import *
from model_factory import get_model
import matplotlib.pyplot as plt
import caption_utils


# Class to encapsulate a neural experiment.
# The boilerplate code to setup the experiment, log stats, checkpoints and plotting have been provided to you.
# You only need to implement the main training logic of your experiment and implement train, val and test methods.
# You are free to modify or restructure the code as per your convenience.
class Experiment(object):
    def __init__(self, name):
        config_data = read_file_in_dir('./', name + '.json')
        if config_data is None:
            raise Exception("Configuration file doesn't exist: ", name)

        self.__name = config_data['experiment_name']
        self.__experiment_dir = os.path.join(ROOT_STATS_DIR, self.__name)

        # Load Datasets
        self.__coco_test, self.__vocab, self.__train_loader, self.__val_loader, self.__test_loader = get_datasets(
            config_data)

        # Setup Experiment
        self.__generation_config = config_data['generation']
        self.__epochs = config_data['experiment']['num_epochs']
        self.__patience = config_data['experiment']['patience']
        self.__current_epoch = 0
        self.__training_losses = []
        self.__val_losses = []
        self.__best_model = None  # Save your best model in this field and use this in test method.

        # Init Model
        self.__model = get_model(config_data, self.__vocab)

        # Criterion and Optimizers set
        self.__criterion = nn.CrossEntropyLoss()
        self.__optimizer = torch.optim.Adam(self.__model.parameters(), lr=0.001)
        # TODO learning rate scheduler??

        self.__init_model()

        # Load Experiment Data if available
        # self.__load_experiment()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loads the experiment data if exists to resume training from last saved checkpoint.
    def __load_experiment(self):
        os.makedirs(ROOT_STATS_DIR, exist_ok=True)

        if os.path.exists(self.__experiment_dir):
            self.__training_losses = read_file_in_dir(self.__experiment_dir, 'training_losses.txt')
            self.__val_losses = read_file_in_dir(self.__experiment_dir, 'val_losses.txt')
            self.__current_epoch = len(self.__training_losses)

            state_dict = torch.load(os.path.join(self.__experiment_dir, 'latest_model.pt'))
            self.__model.load_state_dict(state_dict['model'])
            self.__optimizer.load_state_dict(state_dict['optimizer'])

        else:
            os.makedirs(self.__experiment_dir)

    def __init_model(self):
        if torch.cuda.is_available():
            self.__model = self.__model.cuda().float()
            self.__criterion = self.__criterion.cuda()

    # Main method to run your experiment. Should be self-explanatory.
    def run(self):
        start_epoch = self.__current_epoch
        losses = []
        for epoch in range(start_epoch, self.__epochs):  # loop over the dataset multiple times
            start_time = datetime.now()
            self.__current_epoch = epoch
            train_loss = self.__train()
            val_loss = self.__val()
            losses.append(val_loss)
            print('losses', losses)
            if epoch > self.__patience:
                if losses[-1] > losses[epoch - self.__patience - 1] \
                        and losses[-1] > losses[epoch - self.__patience] \
                        and losses[-1] > losses[epoch - self.__patience + 1]:
                    print('early stopping')
                    break
            self.__record_stats(train_loss, val_loss)
            self.__log_epoch_stats(start_time)
            self.__save_model()

    def __train(self):
        self.__model.train()
        training_loss = 0
        print('train start')

        # Iterate over the data, implement the training function
        for i, (images, captions, _) in enumerate(self.__train_loader):
            self.__optimizer.zero_grad()
            images = images.to(self.device)
            captions = captions.to(self.device)
            outputs = self.__model(images, captions)
            outputs = torch.permute(outputs, (0, 2, 1))
            loss = self.__criterion(outputs, captions)

            loss.backward()
            self.__optimizer.step()

            training_loss += loss.item()
            print(f'iter {i}\tloss {loss.item()}')

        # avg training loss
        training_loss /= len(self.__train_loader)
        return training_loss

    # Perform one Pass on the validation set and return loss value. You may also update your best model here.
    def __val(self):
        self.__model.eval()
        val_loss = 0

        with torch.no_grad():
            for i, (images, captions, img_ids) in enumerate(self.__val_loader):
                images = images.to(self.device)
                captions = captions.to(self.device)

                outputs = self.__model(images, captions)
                outputs = torch.permute(outputs, (0, 2, 1))

                loss = self.__criterion(outputs, captions)

                val_loss += loss.item()

        val_loss /= len(self.__val_loader)
        result_str = "validation Performance: Loss: {} ".format(val_loss)
        self.test()
        self.__log(result_str)

        return val_loss

    # Implement your test function here. Generate sample captions and evaluate loss and
    #  bleu scores using the best model. Use utility functions provided to you in caption_utils.
    #  Note than you'll need image_ids and COCO object in this case to fetch all captions to generate bleu scores.
    def test(self):
        self.__model.eval()
        test_loss = 0
        bleu1 = 0
        bleu4 = 0
        j=0
        with torch.no_grad():
            for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
                images = images.to(self.device)
                captions = captions.to(self.device)

                output = self.__model(images, captions)
                output = torch.permute(output, (0, 2, 1))
                loss = self.__criterion(output, captions)
                test_loss = test_loss + loss
                output = self.__model.predict(images, self.__generation_config["max_length"], self.__generation_config['deterministic'], self.__generation_config['temperature'])
                generated_captions = self.__test_loader.dataset.to_caption(output)
                total_bleu1 = 0
                total_bleu4 = 0
                num_bleu = 0
                for i in range(captions.size()[0]):
                    test_captions = []
                    for annotation in self.__coco_test.imgToAnns[img_ids[i]]:
                        test_caption = annotation['caption']
                        tokenized = nltk.tokenize.word_tokenize(str(test_caption).lower())
                        test_captions.append(tokenized)
                    total_bleu1 += caption_utils.bleu1(test_captions, generated_captions[i])
                    total_bleu4 += caption_utils.bleu4(test_captions, generated_captions[i])
                    num_bleu += 1
                    if j < 5:
                        j = j + 1
                        print("Generated Caption: {}".format(generated_captions[i]))
                        print("Reference Caption: {}".format(test_captions))
                bleu1 += total_bleu1 / num_bleu
                bleu4 += total_bleu4 / num_bleu
        bleu1 /= len(self.__test_loader)
        bleu4 /= len(self.__test_loader)
        test_loss /= len(self.__test_loader)
        result_str = "Test Performance: Loss: {}, Bleu1:{} , Bleu4:{} ".format(test_loss, bleu1, bleu4)

        self.__log(result_str)
        return test_loss, bleu1, bleu4

    def save_3_predictions(self):
        # Get first 3 images from the test loader
        for iter, (images, captions, img_ids) in enumerate(self.__test_loader):
            images_3 = images[:3]
            self.__model.eval()
            images_3 = images_3.to(self.device)
            output = self.__model.predict(images_3, self.__generation_config["max_length"], self.__generation_config['deterministic'], self.__generation_config['temperature'])
            generated_captions = self.__test_loader.dataset.to_caption(output)
        for i in range(3):
            plt.imshow(images_3[i].cpu().numpy().transpose(1, 2, 0))
            plt.title(generated_captions[i])
            plt.savefig(os.path.join(self.__experiment_dir, 'prediction_{}.png'.format(i)))
            plt.show()



    def __save_model(self):
        root_model_path = os.path.join(self.__experiment_dir, 'latest_model.pt')
        model_dict = self.__model.state_dict()
        state_dict = {'model': model_dict, 'optimizer': self.__optimizer.state_dict()}
        torch.save(state_dict, root_model_path)

    def __record_stats(self, train_loss, val_loss):
        self.__training_losses.append(train_loss)
        self.__val_losses.append(val_loss)

        self.plot_stats()

        write_to_file_in_dir(self.__experiment_dir, 'training_losses.txt', self.__training_losses)
        write_to_file_in_dir(self.__experiment_dir, 'val_losses.txt', self.__val_losses)

    def __log(self, log_str, file_name=None):
        print(log_str)
        log_to_file_in_dir(self.__experiment_dir, 'all.log', log_str)
        if file_name is not None:
            log_to_file_in_dir(self.__experiment_dir, file_name, log_str)

    def __log_epoch_stats(self, start_time):
        time_elapsed = datetime.now() - start_time
        time_to_completion = time_elapsed * (self.__epochs - self.__current_epoch - 1)
        train_loss = self.__training_losses[self.__current_epoch]
        val_loss = self.__val_losses[self.__current_epoch]
        summary_str = "Epoch: {}, Train Loss: {}, Val Loss: {}, Took {}, ETA: {}\n"
        summary_str = summary_str.format(self.__current_epoch + 1, train_loss, val_loss, str(time_elapsed),
                                         str(time_to_completion))
        self.__log(summary_str, 'epoch.log')

    def plot_stats(self):
        e = len(self.__training_losses)
        x_axis = np.arange(1, e + 1, 1)
        plt.figure()
        plt.plot(x_axis, self.__training_losses, label="Training Loss")
        plt.plot(x_axis, self.__val_losses, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.legend(loc='best')
        plt.title(self.__name + " Stats Plot")
        plt.savefig(os.path.join(self.__experiment_dir, "stat_plot.png"))
        plt.show()

