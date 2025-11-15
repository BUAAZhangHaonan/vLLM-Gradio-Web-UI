import requests
import json

url = "/compatible-mode/v1/chat/completions"

payload = json.dumps({
   "model": "deepseek-v3",
   "stream": True,
   "messages": [
      {
         "role": "user",
         "content": "9.9和9.11谁大"
      }
   ]
})
headers = {
   'Authorization': 'Bearer API_key', #API_key请填写API密钥
   'Content-Type': 'application/json'
}

response = requests.request("POST", url, headers=headers, data=payload)

print(response.text)