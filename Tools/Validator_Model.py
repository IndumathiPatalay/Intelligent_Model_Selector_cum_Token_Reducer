import requests
import json

url = "https://api.fireworks.ai/inference/v1/chat/completions"
payload = {
  "model": "accounts/fireworks/models/kimi-k2p6",
  "max_tokens": 4096,
  "top_k": 40,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "messages": [
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ]
}
headers = {
  "Accept": "application/json",
  "Content-Type": "application/json",
  "Authorization": "Bearer <FIREWORKS_API_KEY>"
}
requests.request("POST", url, headers=headers, data=json.dumps(payload))