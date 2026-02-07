import json
from openai import OpenAI
import re
import os

class OpenAI_Model:
    def __init__(self, model_name, prompt_prefix, prompt_suffix, prompt_system):
        # load API key
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
        else:
            print("API key not found.")

        # configs
        self.model_name = model_name
        self.prompt_prefix = prompt_prefix
        self.prompt_suffix = prompt_suffix
        self.prompt_system = prompt_system

        self.fallback = -1.

    def format_prompt(self, input: str) -> str:
        return f"{self.prompt_prefix}{input}{self.prompt_suffix}"

    def pred(self, transcript, temperature = 0.0, max_tokens = 512):
        prompt = self.format_prompt(transcript)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.prompt_system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        output = response.choices[0].message.content.strip()
        # try to extract the decimal score after "Rating:"
        match = re.search(
            r"Rating\s*:\s*<?\s*([0-9]+(?:\.\d+)?)\s*>?", 
            output, 
            re.IGNORECASE
        )        
        if match:
            prediction = float(match.group(1))
        else:
            prediction = self.fallback
        
        return prediction