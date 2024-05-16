# MachineTranslationAPI

Welcome to the MachineTranslationAPI repository! Here you will find all the scripts you need to train a powerful machine translation model from scratch, without relying on any pre-trained models. Once trained, you can easily deploy the model as an API, making it accessible for translation tasks. The API is also dockerized, allowing for seamless deployment on any cloud platform. Get ready to unlock the potential of machine translation with this versatile and user-friendly repository!

If you don't want to train the models, there are also available pre-trained models. We provide two versions: 12 SDMT models and a unique Multilingual model.

## Using the API

To use the API, follow these steps:

1. Make sure you have Docker installed on your system.

2. Build the Docker image by running the following command in the terminal:

   ```bash
   docker build -t machine-translation-api .
   ```

   This will create a Docker image named "machine-translation-api" based on the Dockerfile in the repository.

3. Run the Docker container using the following command:

   ```bash
   docker run -p 8000:8000 machine-translation-api
   ```

   This will start the API server on port 8000.

4. Once the API server is running, you can make translation requests using curl or any other HTTP client. Here's an example curl request:

   ```bash
   curl -X POST -H "Content-Type: application/json" -d '{"language_pair": "en-sv", "text": "Hello, world!"}' http://localhost:8000/translate
   ```

   This curl command sends a POST request to the API server with a JSON payload containing the source language, target language, and text to be translated. Adjust the values according to your needs.

   The API will respond with a JSON object containing the translated text.

Remember to replace `http://localhost:8000` with the appropriate URL if you are running the API server on a different host or port.

### Supported Languages

| Language | BLEU (%) |
|----------|----------|
| en-sv    |   2.14   |
| en-da    |   2.52   |
| en-nb    |   7.64   |
| sv-en    |   2.9    |
| sv-nb    |          |
| sv-da    |   5.09   |
| nb-en    |          |
| nb-da    |          |
| nb-sv    |          |
| da-en    |          |
| da-nb    |          |
| da-sv    |          |

## Installation

To install the MachineTranslationAPI, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/MachineTranslationAPI.git
    ```

2. Navigate to the project directory:

    ```bash
    cd MachineTranslationAPI
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Training a Model

To train a machine translation model, follow these steps:

1. Prepare your training data in the desired format.

2. Run the training script:

    ```bash
    python train.py --data_path /path/to/training/data --model_path /path/to/save/model
    ```

    Make sure to replace `/path/to/training/data` with the actual path to your training data and `/path/to/save/model` with the desired path to save the trained model.

3. Monitor the training progress and adjust the hyperparameters as needed.

## Evaluating a Model

To evaluate a trained machine translation model, follow these steps:

1. Prepare your evaluation data in the desired format.

2. Run the evaluation script:

    ```bash
    python evaluate.py --data_path /path/to/evaluation/data --model_path /path/to/saved/model
    ```

    Make sure to replace `/path/to/evaluation/data` with the actual path to your evaluation data and `/path/to/saved/model` with the path to the saved trained model.

3. Analyze the evaluation results and make any necessary improvements to the model.

## Using the API

### Supported Languages

| Language | BLEU (%) |
|----------|----------|
| en-sv    |   2.14   |
| en-da    |   2.52   |
| en-nb    |   7.64   |
| sv-en    |   2.9    |
| sv-nb    |          |
| sv-da    |   5.09   |
| nb-en    |          |
| nb-da    |          |
| nb-sv    |          |
| da-en    |          |
| da-nb    |          |
| da-sv    |          |