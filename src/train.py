import os
import sys
import yaml
import time


import torch

# from torch.nn import functional as F
from torch import optim
from torch.utils.data import DataLoader

# import numpy as np
from tqdm import tqdm

sys.path.append("./src")
from data.tokenizer import MTTokenizer
from data.data_iterator import DataIterator
from models.transformer import Transformer

# from model import SpeakerClassifier
# from loss import *
# from utils import *


class Trainer:
    def __init__(self, params, device):
        self.params = params
        self.device = device
        self._load_data()
        self._load_network()
        self._load_optimizer()
        self._load_criterion()
        self._initialize_training_variables()

    def __load_previous_states(self):
        list_files = os.listdir(self.params["output_directory"])
        list_files = [
            self.params["output_directory"] + "/" + f for f in list_files if ".chkpt" in f
        ]
        if list_files:
            file2load = max(list_files, key=os.path.getctime)
            checkpoint = torch.load(file2load, map_location=self.device)
            try:
                self.net.load_state_dict(checkpoint["model"])
            except RuntimeError:
                self.net.module.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.params = checkpoint["settings"]
            self.starting_epoch = checkpoint["epoch"] + 1
            print('Model "%s" is Loaded for requeue process' % file2load)
        else:
            self.starting_epoch = 1

    def _initialize_training_variables(self):
        if self.params["requeue"]:
            self.__load_previous_states()
        else:
            self.starting_epoch = 0

        self.best_Bleu = 0.0
        self.stopping = 0.0

    def _load_network(self):
        self.net = Transformer(
            source_padding_index=self.tokenizer.source_lang_word_to_id("PAD"),
            target_padding_index=self.tokenizer.target_lang_word_to_id("PAD"),
            target_sos_index=self.tokenizer.target_lang_word_to_id("SOS"),
            encoder_vocabulary_size=len(self.tokenizer.get_source_tokens_dictionary()),
            decoder_vocabulary_size=len(self.tokenizer.get_target_tokens_dictionary()),
            model_dimension=self.params["model_dimension"],
            number_of_heads=self.params["number_of_heads"],
            max_length=self.params["max_length"],
            hidden_dimension=self.params["hidden_dimension"],
            number_of_layers=self.params["number_of_layers"],
            drop_probability=self.params["drop_probability"],
            device=self.device,
        )
        self.net.to(self.device)

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.net = torch.nn.DataParallel(self.net)

    def _load_data(self):
        print("Loading Data for Training")
        self.tokenizer = MTTokenizer()
        self.tokenizer.train(
            self.params["train_src_path"],
            self.params["train_tgt_path"],
            self.params["train_metadata_path"],
            self.params["language_filter_str"],
        )
        data_loader_parameters = {
            "batch_size": self.params["batch_size"],
            "shuffle": True,
            "drop_last": True,
            "num_workers": self.params["num_workers"],
        }
        train_data_iterator = DataIterator(
            self.params["train_src_path"],
            self.params["train_tgt_path"],
            self.params["train_metadata_path"],
            self.params["language_filter_str"],
            self.tokenizer,
        )
        self.params["max_length"] = train_data_iterator.max_source_length
        self.training_generator = DataLoader(
            train_data_iterator, **data_loader_parameters
        )
        self.validation_generator = DataLoader(
            DataIterator(
                self.params["valid_src_path"],
                self.params["valid_tgt_path"],
                self.params["valid_metadata_path"],
                self.params["language_filter_str"],
                self.tokenizer,
            ),
            **data_loader_parameters
        )
       

    def _load_optimizer(self):
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,
                                                 verbose=True,
                                                 factor=self.params["factor"],
                                                 patience=self.params["patience"],)

    def _update_optimizer(self, valid_loss: torch.Tensor):
        if self.epoch > self.params["warmup"]:
            self.scheduler.step(valid_loss)

    def _load_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.target_lang_word_to_id("PAD"))

    def _initialize_batch_variables(self):
        self.train_loss = [None] * len(self.training_generator)
        self.train_batch = 0


    def _calculate_BLEU(self, prediction: torch.Tensor, target_tensor: torch.Tensor, target_length: torch.Tensor) -> float:
        return 0.0


    def _validate(self):
        with torch.no_grad():
            valid_time = time.time()
            valid_loss, batch_BLEU, valid_count = 0.0, list(), 0
            self.net.eval()
            
            for source_tensor, _, target_tensor, target_length in self.validation_generator:
                source_tensor, target_tensor = (
                    source_tensor.long().to(self.device),
                    target_tensor.long().to(self.device),
                )
                prediction = self.net(source_tensor, target_tensor[:,:-1])
                loss = self.criterion(prediction.reshape(-1, prediction.size(-1)), target_tensor[:,1:].reshape(-1))
                valid_loss += loss.item()
                batch_BLEU.append(self._calculate_BLEU(prediction, target_tensor[:,1:], target_length))
                valid_count += 1
           

            BLEU = self.best_Bleu + 1

            print(
                "--Validation Epoch:{epoch: d}, BLEU:{eer: 3.3f}, Loss:{loss: 3.3f}, elapse:{elapse: 3.3f} min".format(
                    epoch=self.epoch, eer=BLEU, loss=valid_loss/valid_count , elapse=(time.time() - valid_time) / 60,
                )
            )
            # early stopping and save the best model
            if BLEU > self.best_Bleu:
                self.best_EER = BLEU
                self.stopping = 0
                print("We found a better model!")
                torch.save({'epoch': self.epoch,
                            'model_state_dict': self.net.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict()},
                             self.params["output_directory"] + "/model.pt")
            else:
                self.stopping += 1

            self.net.train()
            return valid_loss / valid_count

    def _update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

    def __updateTrainningVariables(self, valid_loss):
            self._update_optimizer(valid_loss)

    def _update_metrics(self, loss, batch_looper):
        self.train_loss[self.train_batch] = loss.item()
        self.train_batch += 1
        index_range = slice(
            max(0, self.train_batch - self.params["print_metric_window"]),
            self.train_batch,
        )
        index_len = (
            self.train_batch
            if self.train_batch < self.params["print_metric_window"]
            else self.params["print_metric_window"]
        )
        batch_looper.set_description(
            f"Epoch [{self.epoch}/{self.params['max_epochs']}]"
        )
        batch_looper.set_postfix(
            loss=sum(self.train_loss[index_range]) / index_len
        )

    def train(self):
        print("Start Training")
        for self.epoch in range(
            self.starting_epoch, self.params["max_epochs"]
        ):  # loop over the dataset multiple times
            self.net.train()
            self._initialize_batch_variables()
            batch_looper = tqdm(self.training_generator)
            for source_tensor, _, target_tensor, _ in batch_looper:
                source_tensor, target_tensor = (
                    source_tensor.long().to(self.device),
                    target_tensor.long().to(self.device),
                )
                prediction = self.net(source_tensor, target_tensor[:,:-1])
                loss = self.criterion(prediction.reshape(-1, prediction.size(-1)), target_tensor[:,1:].reshape(-1))
                loss.backward()
                self._update_metrics(loss, batch_looper)
                if self.train_batch % self.params["gradientAccumulation"] == 0:
                     self._update()

            valid_loss = self._validate()

            if self.stopping > self.params["early_stopping"]:
                print("--Best Model EER%%: %.2f" % (self.best_Bleu))
                break

            self.__updateTrainningVariables(valid_loss)

        print("Finished Training")


def main(params):
    print("Defining Device")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))

    print("Loading Trainer")
    trainer = Trainer(params, device)
    trainer.train()


if __name__ == "__main__":

    config_path = sys.argv[1]  # parse input params

    with open(config_path, "rb") as handle:
        params = yaml.load(handle, Loader=yaml.FullLoader)

    if not os.path.exists(params["output_directory"]):
        os.makedirs(params["output_directory"])

    with open(params["output_directory"] + "/config.yaml", "w") as handle:
        yaml.dump(params, stream=handle, default_flow_style=False, sort_keys=False)

    main(params)
