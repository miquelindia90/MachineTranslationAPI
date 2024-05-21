# MachineTranslationAPI

Welcome to the MachineTranslationAPI repository! Here you will find all the scripts you need to train a set of neural machine translation models from scratch, without relying on any pre-trained models. Once trained, you can easily deploy these models as an API, making it accessible for translation tasks. The API is also dockerized, allowing for seamless deployment on any cloud platform. Any doubt or question, please feel free to contact me or write any issue in the repository.

If you don't want to train the models, there are also available a collection of pre-trained models. We currently provide a set of 12 Single Direction Machine Translation (SDMT) models. I'm working on a second version of a multilingual model, that will be available soon.

## Using the API

To use the API, follow these steps:

0. Make sure models are correctly downloaded in `models/` directory. These models are large so you need to use `git lfs` to have them downloaded in its original form (not the lfs pointer). Their size is around 300-500 MB. So if they don't have the correct size, you need to fix them. You can convert them using the following command:

   ```bash
   git lfs pull
   ```

   This will download the pre-trained models in their correct format in the `models/` directory.

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

   This curl command sends a `POST` request to the API server with a JSON payload containing the language pair (`src-tgt`) and the text to be translated. Adjust the values according to your needs.

   The API will respond with a JSON object containing the translated text.

If you wanted to use the API without the Docker container, you can run the following commands:

1. Install first the required dependencies by running the following command in the terminal:

    ```bash
    pip install -r requirements.txt
    ```
    The repository supports python=[3.8, 3.9, 3.10]

2. Run the API server using the following command:

    ```bash
    python api.py
    ```
    By default the API will be run on localhost:5000
    
## Supported Languages

Here are the languages supported by the MachineTranslationAPI. The table below shows the BLEU scores for the individual SDMT models. 

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
      - Additionally, you can have a third metadata file that contains information about the source and target pairs. A variable called `language_filter_str` can be used during training to filter the data based on the metadata.

      This structure follows LanguageWire public [data](https://languagewire-my.sharepoint.com/personal/adas_languagewire_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fadas%5Flanguagewire%5Fcom%2FDocuments%2Fsenior%5Fml%5Fengineer%5Ftech%5Fchallenge%5Fdata%5Flw%5Fmlt%5Ftech%5Fchallenge%2Ezip&parent=%2Fpersonal%2Fadas%5Flanguagewire%5Fcom%2FDocuments&ga=1) format. If you have data in a different format, you will need to preprocess it to match this structure.
      

2. Run the training script:

   ```bash
   python train.py configs/config.yaml
   ```

   This command runs the whole training process. All the variables needed for the training as data paths, hyperparameters, and model sizes are defined in the config.yaml file. You can modify this file to adjust the training process to your needs.

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

   This script will evaluate the model using the test data and save the translation results in the output_path. You can use whether if you want o use a cpu "cpu" or a gpu "cuda:0" to do the evaluation. If the cpu choice is made, the evaluation will be a bit slower, despite using dynamic quantization with int8. The result metric BLEU will be displayed in the terminal, at the end of the evaluation.

## Findings

This repo has been tested using the [dataset](https://languagewire-my.sharepoint.com/personal/adas_languagewire_com/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fadas%5Flanguagewire%5Fcom%2FDocuments%2Fsenior%5Fml%5Fengineer%5Ftech%5Fchallenge%5Fdata%5Flw%5Fmlt%5Ftech%5Fchallenge%2Ezip&parent=%2Fpersonal%2Fadas%5Flanguagewire%5Fcom%2FDocuments&ga=1) provided by LanguageWire. Here are some findings, improvements and future work that could be done in this repository:

1.- **Data Pre-Processing:** The data was pre-cleaned but there was some work to do yet. We needed to implement the following modules/steps:

1.1 *Normalization*: It was needed to normalize the text, lowercasing the sentences and separating punctuation and other special charcaters from words. There were some special characters that I didn't know how to deal with. I preferred initally to process them as they were punctuation marks, but it could have been a good idea to remove them. The capitalized text was also a bit complex issue, because looking in the sentences I noticed that some non-initial sentecene words have capital letters that could provide some linguistic information. I decided to lowercased all the text, but maybe it could have been a good idea to keep the capital letters in the non-initial words.

1.2 *Tokenization*: The tokenization was implemented without using any external library. The tokenization was done in a way that the sentences were tokenized in words. In SDMT models we have worked in the word level. For future SDMT or multilingual models I would like to work with subwords ([BPE](https://aclanthology.org/P16-1162/) or [Unigram LM](https://arxiv.org/abs/1804.10959)) in order to avoid dealing with a almost ~4x vocabulary size (multilingual) and also with the OOV issue. 

2.- **Algorithm Design**: The first idea was to train a collection of SDMT models that should be used in a multilingual platform API. Since I was not able to run thousand of trainings I had to choose whether to use a LAS based topology or a Transformer based one. I chose the Tranformer one and tried to run a first set of small models with a small Transformer configuration. A second set of trainings was done playing with some configuration variable changes. This second set of trainings did not lead to a noticeable improvement in terms of BLEU.

3.- **Results and Discussion**: The resuls table with the BLEU metric for each language pair can be found in [Supported Languages](#supported-languages). After the training + evaluation steps, the results were not good. Here is a list of observations and learnings made during the process.

* It seems 150k samples per language pair is not enough to train a good SDMT model with a Transformer.
* Maybe LAS could have worked better than the Transformer given this ammount of data.
* Subword tokenization could haved lead to a better BLEU in average. 
* It seems a good idea to train a multilingual model, but I'm not sure if even with that the amount of data would be enough to train good models. Subword tokenization and using Language Tokens to identify the source/tgt language pair would be a nice approach to start with. 
* It is needed to play more with some training hyperparameters.
* It is needed to play more with the decoding (Beam Search) variables.
* It is needed to analyze more the reason of why some language-pair models have shown better results than others. Nb pairs have shown better results than the other combinations.
* It is needed to have a post processing system to correct some output tokenization issues.
* Maybe starting from a pretained model and using a robust fine tuning method like a LORA based one could have shortened and improved this whole process. 

4.- **Training/Inference Optimization**:

Here are some observations and methods that could help to improve both training and inference steps.

* Better to train a 100 multilingual model than training 100x99 SDMT individual ones. This scales better in several cases metrics in training speed or total model size but mabe not in inference speed. Additionally, It would be a challenge when It was needed to train models with very different character dictionaries like latin + kanjis.

* Inference optimization can be done with different libraries (Torch_Script, ONNX Runtime, TensorRT) and methods (static/dynamic quantization, weight pruning). If GPUS are available for production, channel multiplexation with GPU batching could also be considered. 

5.- **Production Technical Considerations**:

Here are few things that I think should be considered for moving some models to production:

* SDMT approach works but seems inefficient. You could be having some models in RAM not being used meanwhile some other models have queded petitions because they are receiving more traffic.
* Multilingual approach seems to be very efficient but if the model size is not very large. Overscaling the number of languages seems not a good idea if you are not able to have at least a X ammount of models concurrently working. Hence there is a trade-off that must be considered.
* Not all the work comes from the AI. A good output text formatter could fix some model bad design issues and even handle some model uncorrect behaviors. 

6.- **Improvement ROADMAP**:

This is the lisf ot the things that I think that should be done in order to obtain the final multilingual MT model: 

* (Short Term): Implement a robust output text formatter.
* (Short Term): Achieve a better set of SDMT models varying model/training hyperparameters and beam serach variable optimizaiton.
* (Short Term): Analyse Pre-Trained model + Fine Tuning approaches.
* (Mid Term): Implement and train a  first Multilingual approach (4 languages).
* (Mid Term): Explore different topologies, configurations for Multlingual models.
* (Mid Term): Explore how to optimize multilingual inference.
* (Long Term): Explore how to scale to 100 models.
