
import sys

sys.path.append("./src")

from inference.translation_platform import MultiLanguageTranslator

def test_MultiLanguageTranslator_initialization():
    translator = MultiLanguageTranslator(models_directory="models/")
    assert "en-sv" in translator.translators

def test_MultiLanguageTranslator_translation():
    translator = MultiLanguageTranslator(models_directory="models/")

    translation = translator.translate("This is a test sentence.", "en-sv")
    assert translation == "• om mening ."
    
if __name__ == "__main__":
    test_MultiLanguageTranslator_initialization()
    test_MultiLanguageTranslator_translation()
