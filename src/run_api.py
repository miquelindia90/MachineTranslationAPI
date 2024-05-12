from flask import Flask, request, jsonify
from your_translation_module import Translator

app = Flask(__name__)
translador = Translador()

@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.json
    input_text = data['text']
    target_language = data['target_language']
    translated_text = translator.translate(input_text, target_language)
    return jsonify({'translated_text': translated_text})

if __name__ == '__main__':
    app.run(debug=True)