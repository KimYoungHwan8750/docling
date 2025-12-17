import base64
import requests
from openai import OpenAI

class QwenClient:
    def __init__(self, ip="192.168.0.99", port="8000", model="My_Model"):
        self.api_url = f"http://{ip}:{port}/v1"
        self.model = model
        self.client = OpenAI(base_url=self.api_url, api_key="EMPTY")

    def ask_text(self, prompt, max_tokens=512):
        """텍스트 전용 질문 메서드"""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.7
        )
        return response.choices[0].message.content

    def ask_image(self, prompt, image_path, max_tokens=1024):
        """이미지 분석 질문 메서드"""
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            "max_tokens": max_tokens
        }
        response = requests.post(f"{self.api_url}/chat/completions", json=payload)
        return response.json()['choices'][0]['message']['content']