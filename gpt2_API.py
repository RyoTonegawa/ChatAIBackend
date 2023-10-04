from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# download_gpt2_script.pyでダウンロードしたモデルとトークナイザーをロード
tokenizer = GPT2Tokenizer.from_pretrained('./model_cache/gpt2')
model = GPT2LMHeadModel.from_pretrained('./model_cache/gpt2')

@app.route('/predict',methods = ['POST'])
def predict():
    # テキストのパース
    text = request.json['text']
    inputs = tokenizer.encode(text, return_tensors='pt')

    # モデルの実行
    with torch.no_grad():
        outputs = model.generate(inputs, max_length=150, num_beams=5, temperature=1.5, top_k=50)
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 出力の取得＆返却
    print(jsonify({'result': response_text}))
    return jsonify({'result': response_text})

if __name__ == '__main__':
    app.run(port=5000)
