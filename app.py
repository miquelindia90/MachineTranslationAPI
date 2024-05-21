import sys
import json

from flask import Flask, request

sys.path.append("src/")
from inference.translation_platform import MultiLanguageTranslator

app = Flask(__name__)
translator = MultiLanguageTranslator("models/")


@app.route("/translate", methods=["POST"])
def translate_text():
    data = request.json
    input_text = data["text"]
    language_pair = data["language_pair"]
    translated_text = translator.translate(input_text, language_pair)
    response = {"translated_text": translated_text}
    return app.response_class(
        response=json.dumps(response, ensure_ascii=False),
        mimetype='application/json'
    )

if __name__ == "__main__":
    app.run(debug=True)
