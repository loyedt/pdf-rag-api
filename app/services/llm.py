import os
import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

def generate_answer(prompt):
    res = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": "qwen2.5:0.5b",
            "prompt": prompt,
            "stream": False
        }
    )
    return res.json()["response"]