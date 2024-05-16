import os
import sys
import glob

import yaml

sys.path.append("./src")

from inference.inference_utils import prepare_translator

class MultiLanguageTranslator:
    """
    A class that represents a multi-language translator.

    Attributes:
        models_directory (str): The directory path where the translation models are stored.
        translators (dict): A dictionary that maps language codes to translator objects.
    """

    def __init__(self, models_directory):
        self.models_directory = models_directory
        self._prepare_translators()

    def _prepare_translators(self):
        """
        Prepares the translators by loading the models and creating translator objects.
        """
        self.translators = {}
        model_paths = glob.glob(self.models_directory + "/*")
        for model_path in model_paths:
            model_parameters = yaml.safe_load(open(model_path + "/config.yaml", "r"))
            self.translators[os.path.basename(model_path)] = prepare_translator(model_parameters, model_path)

    def translate(self, sentence, language_code):
        """
        Translates a sentence to the specified language.

        Args:
            sentence (str): The sentence to be translated.
            language_code (str): The language code of the target language.

        Returns:
            str: The translated sentence.
        """
        return self.translators[language_code].translate(sentence)