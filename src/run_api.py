import sys

from flask import Flask, request, jsonify

sys.path.append("src/")
from inference.translation_platform import MultiLanguageTranslator

app = Flask(__name__)
translator = MultiLanguageTranslator("models/")


@app.route("/translate", methods=["POST"])
def translate_text():
    data = request.json
    input_text = data["text"]
    target_language = data["target_language"]
    translated_text = translator.translate(input_text, target_language)
    return jsonify({"translated_text": translated_text})


if __name__ == "__main__":
    app.run(debug=True)
