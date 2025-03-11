import json
import re

import boto3
from openai import OpenAI
import torch
from transformers import pipeline

from keys import OPENAI_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY


class CodeGenerationLLM:
    def __init__(self, temperature: float, top_p: float):
        self.temperature = temperature
        self.top_p = top_p

    def generate_code(self, prompt, max_new_tokens):
        raise NotImplementedError("This method should be implemented by subclasses.")


class HFModel(CodeGenerationLLM):
    def __init__(self, temperature: float, top_p: float, model_name: str, torch_dtype=torch.float16):
        super().__init__(temperature, top_p)
        self.model_name = model_name
        self.pipeline = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    @staticmethod
    def build_message(role: str, content: str):
        return {"role": role, "content": content}

    @staticmethod
    def build_messages_from_task(model_name: str, system: str, content: str):
        if 'gemma' in model_name:
            messages = [HFModel.build_message("user", f"{system}\n\n{content}")]
        elif 'starchat' in model_name:
            prompt_template = "<|system|>\n{system}<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>"
            messages = prompt_template.format(system=system, query=content)
        else:
            messages = []
            if system:
                messages.append(HFModel.build_message("system", system))
            messages.append(HFModel.build_message("user", content))

        return messages

    def generate_code(self, system: str, content: str, max_new_tokens: int = 2 ** 12):
        with torch.no_grad():
            # Build the message JSON
            messages = HFModel.build_messages_from_task(self.model_name, system, content)

            # Generate text using the pipeline
            additional_args = {"eos_token_id": 49155} if 'starchat' in self.model_name else {}
            outputs = self.pipeline(
                messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                **additional_args
            )
        return outputs[0]["generated_text"][-1]["content"] if 'starchat' not in self.model_name else \
            outputs[0]["generated_text"].split('<|assistant|>')[-1].strip()


class GPT(CodeGenerationLLM):
    def __init__(self, temperature: float, top_p: float, model_name: str, **kwargs):
        super().__init__(temperature, top_p)
        self.model_name = model_name
        self.api_key = OPENAI_API_KEY
        self.client = OpenAI(api_key=self.api_key)

    def generate_code(self, system: str, content: str, max_new_tokens: int = 2 ** 12):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_completion_tokens=max_new_tokens,
            n=1
        )
        return response.choices[0].message.content

    def get_model_name(self):
        raise NotImplementedError()


class Claude3Haiku(CodeGenerationLLM):
    HAIKU_ID = "anthropic.claude-3-haiku-20240307-v1:0"

    def __init__(self, temperature: float, top_p: float, **kwargs):
        super().__init__(temperature, top_p)
        session = boto3.Session(aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        self.bedrock_client = session.client(
            service_name='bedrock-runtime',
            region_name='us-east-1',
            endpoint_url=f'https://bedrock-runtime.us-east-1.amazonaws.com',
        )

    def generate_code(self, system: str, content: str, max_new_tokens: int = 2 ** 12):
        system_dict = {}
        if system:
            system_dict["system"] = system

        response = self.bedrock_client.invoke_model(
            modelId=Claude3Haiku.HAIKU_ID,
            contentType="application/json",
            accept="*/*",
            body=json.dumps({
                **system_dict,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": content}
                    ]
                }],
                "max_tokens": max_new_tokens,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'anthropic_version': 'bedrock-2023-05-31',
            })
        )

        return json.loads(response['body'].read())['content'][0]['text']


class ResponseSanitizer:
    @staticmethod
    def sanitize_response(response, model_name, language='python'):
        # Keep only the code, remove other irrelevant information
        if language == 'python':
            matches = re.finditer(r"```(?:py(?:thon)?)?\n(.*?)```", response, re.DOTALL)
        elif language == 'java':
            matches = re.finditer(r"```(?:java)?\n(.*?)```", response, re.DOTALL)
        elif language == 'cpp':
            matches = re.finditer(r"```(?:cpp|c\+\+|cxx)?\n(.*?)```", response, re.DOTALL)
        else:
            raise ValueError("Unsupported language")

        matches = tuple(matches)

        if len(matches) == 1:
            # Extract the code snippet inside the triple backticks
            response = matches[0].group(1).strip()
        else:
            response = ""

        for prefix in ['# Test', '# Example']:
            response = response.split(prefix)[0]

        if 'gpt-4o' in model_name:
            # GPT-4o sometimes generates irrelevant example code, remove it
            irrelevant_code_prefixes = ['# Input and output', '# Input reading', '# Input Reading',
                                        '# Read input', '# Read Input', '# Reading input', '# Reading Input',
                                        '# Read the input', '# Read the Input']
            for prefix in irrelevant_code_prefixes:
                if prefix in response:
                    split = response.split(prefix, maxsplit=1)
                    if len(split[0]) > 0.5 * len(split[1]):
                        response = split[0].strip()
            response = '\n'.join([x for x in response.splitlines() if not any(x.strip().startswith(p) for p in irrelevant_code_prefixes)])
            if response.endswith('\n'):
                response = response.rstrip('\n') + '\n'

        if not ResponseSanitizer.is_valid_function(response):
            response = ""

        return response

    @staticmethod
    def is_valid_function(snippet):
        lines = [line.strip() for line in snippet.strip().split('\n') if line.strip()]

        if not lines or not lines[0].startswith("def "):
            return True  # If it's not a function, consider it valid

        if len(lines) < 2:
            return False  # A valid function must have at least a definition line and a body

        body_lines = lines[1:]
        valid_body = not all(line.strip().startswith("#") or line.strip() == "pass" for line in body_lines)
        return valid_body
