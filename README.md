# MachineTranslationAPI


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

| Language | BLEU |
|----------|------|
| en-sv    |      |
| en-da    |      |
| en-nb    |      |
| sv-en    |      |
| sv-nb    |      |
| sv-da    |      |
| nb-en    |      |
| nb-da    |      |
| nb-sv    |      |
| da-en    |      |
| da-nb    |      |
| da-sv    |      |