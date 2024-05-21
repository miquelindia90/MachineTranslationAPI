# MachineTranslationAPI

Welcome to the MachineTranslationAPI repository! Here you will find all the scripts you need to train neural machine translation models from scratch, without relying on any pre-trained models. Once trained, you can easily deploy these models as an API, making it accessible for translation tasks. The API is also dockerized, allowing for seamless deployment on any cloud platform. Any doubt or question, please feel free to contact me or write any issue in the repository.

If you don't want to train the models, there are also available pre-trained models. We currently provide a set of 12 SDMT models. I'm working on a second version of a multilingual model, that will be available soon.

## Using the API

To use the API, follow these steps:

0. Make sure models are correctly downloaded in `models/` directory. These models are large so you need to use `git lfs` to have them downloaded in its original form (not the lfs pointer). They size are around 300-500 MB. So if they don't have the correct size, you need to fix them. You can convert them using the following command:

   ```bash
   git lfs pull
   ```

   This will download the pre-trained models in its correct format in the `models/` directory.

1. Make sure you have [Docker](https://www.docker.com/) installed on your system.

2. Build the Docker image by running the following command in the terminal:

   ```bash
   docker build -t machine-translation-api .
   ```

   This will create a Docker image named `machine-translation-api` based on the Dockerfile in the repository.

3. Run the Docker container using the following command:

   ```bash
   docker run -p <HOST_PORT>:5000 machine-translation-api
   ```

   This will start the API server on port <HOST_PORT>.

4. Once the API server is running, you can make translation requests using curl or any other HTTP client. Here's an example curl request:

   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"language_pair": "en-sv", "text": "Hello, world!"}' http://localhost:<HOST_PORT>/translate
   ```

   This curl command sends a `POST` request to the API server with a JSON payload containing the language pair (`src-tgt`) and text to be translated. Adjust the values according to your needs.

   The API will respond with a JSON object containing the translated text.

If you wanted to use the API without the Docker container, you can run the following commands:

1. Install the required dependencies by running the following command in the terminal:

    ```bash
    pip install -r requirements.txt
    ```
    The repository supports python=[3.8, 3.9, 3.10]

2. Run the API server using the following command:

    ```bash
    python3 api.py
    ```
    By default the API is run on localhost:5000
    
### Supported Languages

Here are the languages supported by the MachineTranslationAPI. The table below shows the BLEU scores for the individual single direction models (SDMT). 

| Language | SDMT BLEU (%) |
|:--------:|:-------------:|
| en-sv    |     2.14      |
| en-da    |     2.52      |
| en-nb    |     7.64      |
| sv-en    |     2.9       |
| sv-nb    |     6.02      |
| sv-da    |     5.09      |
| nb-en    |     7.58      |
| nb-da    |     3.62      |
| nb-sv    |     5.35      |
| da-en    |     0.95      |
| da-nb    |     4.71      |
| da-sv    |     3.2       |

## Training a Model

To train a machine translation model, follow these steps:

0. Install the required dependencies by running the following command in the terminal:

   ```bash
   pip install -r requirements.txt
   ```
   This will install all the necessary packages specified in the `requirements.txt` file. Make sure you have Python 3.8, 3.9, or 3.10 installed on your system as the repository supports these versions.

1. Prepare your training data in the following format:

      - You need to have data for the three common data splits: training, validation, and test.
      - For each split you need at least two files: one for the source language and one for the target language. This files must contain the sentences in the corresponding language, one sentence per line.
      - Additionally, you can have a third metadata file that contains information about the source and target pairs. A variable called language_filter_str can be used during training to filter the data based on the metadata.

      This structure follows LanguageWire public [data]{https://languagewire-my.sharepoint.com/personal/adas_languagewire_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fadas%5Flanguagewire%5Fcom%2FDocuments%2Fsenior%5Fml%5Fengineer%5Ftech%5Fchallenge%5Fdata%5Flw%5Fmlt%5Ftech%5Fchallenge%2Ezip&parent=%2Fpersonal%2Fadas%5Flanguagewire%5Fcom%2FDocuments&ga=1} format. If you have data in a different format, you will need to preprocess it to match this structure.
      

2. Run the training script:

   ```bash
   python train.py configs/config.yaml
   ```

   This command runs the whole training process. All the variables needed for the training as data_path s, hyperparameters, and model architecture are defined in the config.yaml file. You can modify this file to adjust the training process to your needs.

   After the training is complete, the script will save the trained model to the specified output directory. You will find the following files in the output directory:
      - config.yaml: A copy of the configuration file used for training.
      - model.pth: The trained model weights.
      - source_tokens.json and target_tokens.json: Tokenizers for the source and target languages.

   Once you have the model trained you can evaluate it using the evaluation script or if you want to use it in production you can deploy it as an API.

## Evaluating a Model

Before anything, make sure that you have installed the corresponding python dependencies (explained in previous section step 0). To evaluate a trained machine translation model, follow these steps:

1. Run the evaluation script:

    ```bash
    python evaluate_model.py -m <model_path> -o <output_path> -d cpu
    ```

   This script will evaluate the model using the test data and save the translation results in the output_path. You can use wether if you want o use a cpu "cpu" or a gpu "cuda:0" to do the evaluation. If the cpu choice is made, the evaluation will be a bit slower, despit using dynamic quantization with int8. The result metric BLEU will be displayed in the terminal, at the end of the evaluation.

## Findings

This repo has been tested using the dataset provided by LanguageWire in [here]{https://languagewire-my.sharepoint.com/personal/adas_languagewire_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fadas%5Flanguagewire%5Fcom%2FDocuments%2Fsenior%5Fml%5Fengineer%5Ftech%5Fchallenge%5Fdata%5Flw%5Fmlt%5Ftech%5Fchallenge%2Ezip&parent=%2Fpersonal%2Fadas%5Flanguagewire%5Fcom%2FDocuments&ga=1}. Here are some findings, improvements and future work that could be done in this repository:

1.- Data Processing: The data was pre-cleaned but still there was some work to do. We needed to implement the following steps:

      - Normalization: It was needed to normalize the text, lowercasing the sentences and separating punctuation and other special charcaters from words. There were some special charaters that I didn't know to deal with. I prefered initally to let them be as punctuaion characters, but it could be a good idea to remove them. The lowercased text was also a bit delicated process, because looking in the sentences I noticed that some non-initial sentecene words have capital letters that could provide some linguistic information. I decided to lowercased all the text, but maybe it could have been a good idea to keep the capital letters in the non-initial words.

      - Tokenization: The tokenization was done without using any external library. The tokenization was done in a way that the sentences were tokenized in words. In SDMT models we have worked in the word level. In the multilingual case we have worked with subwords in order to avoid dealing with a almos ~4x vocabulary size and also with the OOV issue.


