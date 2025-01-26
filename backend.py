# This code is a telegram chatbot based on the following open source LLM model.
# https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import pipeline
from flask import Flask, request
import requests

BOT_TOKEN = "xxx" # FIXME: Please use your own token
TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.get_json().get('message')
    chat_id = data.get('chat').get('id')
    text = data.get('text')

    # LLM
    pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {"role": "system", "content": "You are a friendly chatbot.",},
        {"role": "user", "content": text},
    ]
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

    # Send the response back to the user
    reply_url = f"{TELEGRAM_API}/sendMessage"
    params = {
        "chat_id": chat_id,
        "text": outputs[0]["generated_text"].split('<|assistant|>\n')[1]
    }
    requests.post(reply_url, json=params)

    return "OK", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2345, debug=True)
