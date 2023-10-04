from transformers import GPT2Tokenizer, GPT2LMHeadModel

# オンラインでモデルとトークナイザーをダウンロード
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# モデルとトークナイザーをローカルに保存
tokenizer.save_pretrained('./model_cache/gpt2')
model.save_pretrained('./model_cache/gpt2')
