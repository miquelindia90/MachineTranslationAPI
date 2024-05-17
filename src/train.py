import os
import sys
import yaml
import time


import torch
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm

sys.path.append("./src")
from data.data_utils import write_translation_output
from data.tokenizer import MTTokenizer
from data.data_iterator import DataIterator
from data.scoring import calculate_batch_bleu_score
from models.transformer import Transformer


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
            self.params["output_directory"] + "/" + f
            for f in list_files
            if ".chkpt" in f
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

        self.best_valid_loss = 1_000
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
        self.tokenizer.save_tokens_dictionary(
            self.params["output_directory"] + "/source_tokens.json",
            self.params["output_directory"] + "/target_tokens.json",
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
        self.training_generator = DataLoader(
            train_data_iterator, **data_loader_parameters
        )
        data_loader_parameters = {
            "batch_size": self.params["batch_size"],
            "shuffle": False,
            "drop_last": False,
            "num_workers": self.params["num_workers"],
        }
        self.validation_generator = DataLoader(
            DataIterator(
                self.params["valid_src_path"],
                self.params["valid_tgt_path"],
                self.params["valid_metadata_path"],
                self.params["language_filter_str"],
                self.tokenizer,
            ),
            **data_loader_parameters,
        )
        self.test_iterator = DataIterator(
            self.params["test_src_path"],
            self.params["test_tgt_path"],
            self.params["test_metadata_path"],
            self.params["language_filter_str"],
            self.tokenizer,
        )
        self.test_generator = DataLoader(self.test_iterator, **data_loader_parameters,)

    def _load_optimizer(self):
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=self.params["learning_rate"],
            weight_decay=self.params["weight_decay"],
        )

    def _load_criterion(self):
        self.criterion = torch.nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.target_lang_word_to_id("PAD")
        )

    def _initialize_batch_variables(self):
        self.train_loss = [None] * len(self.training_generator)
        self.train_batch = 0

    def _calculate_valid_loss(self):
        valid_time = time.time()
        valid_loss, valid_count = 0.0, 0

        for source_tensor, _, target_tensor, _ in self.validation_generator:
            source_tensor, target_tensor = (
                source_tensor.long().to(self.device),
                target_tensor.long().to(self.device),
            )
            prediction = self.net(source_tensor, target_tensor[:, :-1])
            loss = self.criterion(
                prediction.reshape(-1, prediction.size(-1)),
                target_tensor[:, 1:].reshape(-1),
            )
            valid_loss += loss.item()
            valid_count += 1

        return valid_loss / valid_count, time.time() - valid_time

    def _write_epoch_translations(self, predictions: list):

        source_sentences = list()
        target_sentences = list()
        translated_sentences = list()

        for index, prediction in enumerate(predictions):
            source_sentences.append(self.test_iterator.get_source_sentence(index))
            target_sentences.append(self.test_iterator.get_target_sentence(index))
            translated_sentence = self.tokenizer.target_lang_list_id_to_word_list(
                prediction
            )
            if translated_sentence[-1] == self.tokenizer.source_lang_word_to_id("EOS"):
                translated_sentence = translated_sentence[:-1]
            translated_sentences.append(" ".join(translated_sentence))

        write_translation_output(
            source_sentences,
            translated_sentences,
            target_sentences,
            self.params["output_directory"] + "/translations.txt",
        )

    def _evaluate_assisted_bleu(self):
        assisted_bleu = 0.0
        batch_count = 0.0
        predictions = list()
        for source_tensor, _, target_tensor, target_length in self.test_generator:
            batch_count += 1
            source_tensor, target_tensor = (
                source_tensor.long().to(self.device),
                target_tensor.long().to(self.device),
            )
            prediction = self.net(source_tensor, target_tensor[:, :-1])
            predictions += [
                torch.argmax(prediction[index], dim=-1)
                .squeeze()
                .tolist()[: target_length[index].item()]
                for index in range(prediction.size()[0])
            ]
            assisted_bleu += calculate_batch_bleu_score(
                prediction, target_tensor[:, 1:], target_length, self.tokenizer
            )
        self._write_epoch_translations(predictions)
        return assisted_bleu / batch_count

    def _validate(self):
        with torch.no_grad():
            self.net.eval()
            valid_loss, valid_elpased_time = self._calculate_valid_loss()
            assisted_bleu = self._evaluate_assisted_bleu()
            print(
                "--Validation Epoch:{epoch: d}, Loss:{loss: 3.3f}, Assisted_BLEU:{bleu: 3.3f} elapse:{elapse: 3.3f} min".format(
                    epoch=self.epoch,
                    loss=valid_loss,
                    bleu=assisted_bleu,
                    elapse=valid_elpased_time / 60,
                )
            )
            # early stopping and save the best model
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.stopping = 0
                print("We found a better model!")
                torch.save(
                    {
                        "epoch": self.epoch,
                        "model_state_dict": self.net.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                    },
                    self.params["output_directory"] + "/model.pt",
                )
            else:
                self.stopping += 1

            self.net.train()

    def _update(self):
        self.optimizer.step()
        self.optimizer.zero_grad()

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
        batch_looper.set_postfix(loss=sum(self.train_loss[index_range]) / index_len)

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
                prediction = self.net(source_tensor, target_tensor[:, :-1])
                loss = self.criterion(
                    prediction.reshape(-1, prediction.size(-1)),
                    target_tensor[:, 1:].reshape(-1),
                )
                loss.backward()
                self._update_metrics(loss, batch_looper)
                if self.train_batch % self.params["gradientAccumulation"] == 0:
                    self._update()

            self._validate()

            if self.stopping > self.params["early_stopping"]:
                print("--Best Model Valid Loss%%: %.2f" % (self.best_valid_loss))
                break

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
