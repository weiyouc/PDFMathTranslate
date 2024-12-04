import html
import logging
import os
import re
from json import dumps, loads

import deepl
import ollama
import openai
import requests
from azure.ai.translation.text import TextTranslationClient
from azure.core.credentials import AzureKeyCredential


class BaseTranslator:
    def __init__(self, service, lang_out, lang_in, model):
        self.service = service
        self.lang_out = lang_out
        self.lang_in = lang_in
        self.model = model

    def translate(self, text) -> str:
        ...

    def __str__(self):
        pass

    def __str__(self):
        return f"{self.service} {self.lang_out} {self.lang_in}"


class GoogleTranslator(BaseTranslator):
    def __init__(self, service, lang_out, lang_in, model):
        lang_out = "zh-CN" if lang_out == "auto" else lang_out
        lang_in = "en" if lang_in == "auto" else lang_in
        super().__init__(service, lang_out, lang_in, model)
        self.session = requests.Session()
        self.base_link = "http://translate.google.com/m"
        self.headers = {
            "User-Agent": "Mozilla/4.0 (compatible;MSIE 6.0;Windows NT 5.1;SV1;.NET CLR 1.1.4322;.NET CLR 2.0.50727;.NET CLR 3.0.04506.30)"
        }

    def translate(self, text):
        text = text[:5000]  # google translate max length
        response = self.session.get(
            self.base_link,
            params={"tl": self.lang_out, "sl": self.lang_in, "q": text},
            headers=self.headers,
        )
        re_result = re.findall(
            r'(?s)class="(?:t0|result-container)">(.*?)<', response.text
        )
        if len(re_result) == 0:
            raise ValueError("Empty translation result")
        else:
            result = html.unescape(re_result[0])
        return result


class DeepLXTranslator(BaseTranslator):
    def __init__(self, service, lang_out, lang_in, model):
        lang_out = "zh" if lang_out == "auto" else lang_out
        lang_in = "en" if lang_in == "auto" else lang_in
        super().__init__(service, lang_out, lang_in, model)
        try:
            auth_key = os.getenv("DEEPLX_AUTH_KEY")
            server_url = (
                "https://api.deeplx.org"
                if not os.getenv("DEEPLX_SERVER_URL")
                else os.getenv("DEEPLX_SERVER_URL")
            )
        except KeyError as e:
            missing_var = e.args[0]
            raise ValueError(
                f"The environment variable '{missing_var}' is required but not set."
            ) from e

        self.session = requests.Session()
        self.base_link = f"{server_url}/{auth_key}/translate"
        self.headers = {
            "User-Agent": "Mozilla/4.0 (compatible;MSIE 6.0;Windows NT 5.1;SV1;.NET CLR 1.1.4322;.NET CLR 2.0.50727;.NET CLR 3.0.04506.30)"
        }

    def translate(self, text):
        text = text[:5000]  # google translate max length
        response = self.session.post(
            self.base_link,
            dumps(
                {
                    "target_lang": self.lang_out,
                    "text": text,
                }
            ),
            headers=self.headers,
        )
        # 1. Status code test
        if response.status_code == 200:
            result = loads(response.text)
        else:
            raise ValueError("HTTP error: " + str(response.status_code))
        # 2. Result test
        try:
            result = result["data"]
            return result
        except KeyError:
            result = ""
            raise ValueError("No valid key in DeepLX's response")
        # 3. Result length check
        if len(result) == 0:
            raise ValueError("Empty translation result")
        return result


class DeepLTranslator(BaseTranslator):
    def __init__(self, service, lang_out, lang_in, model):
        lang_out='ZH' if lang_out=='auto' else lang_out
        lang_in='EN' if lang_in=='auto' else lang_in
        super().__init__(service, lang_out, lang_in, model)
        self.session = requests.Session()
        auth_key = os.getenv('DEEPL_AUTH_KEY')
        server_url = os.getenv('DEEPL_SERVER_URL')
        self.client = deepl.Translator(auth_key, server_url=server_url)

    def translate(self, text):
        response = self.client.translate_text(
            text,
            target_lang=self.lang_out,
            source_lang=self.lang_in
        )
        return response.text


class OllamaTranslator(BaseTranslator):
    def __init__(self, service, lang_out, lang_in, model):
        lang_out='zh-CN' if lang_out=='auto' else lang_out
        lang_in='en' if lang_in=='auto' else lang_in
        super().__init__(service, lang_out, lang_in, model)
        self.options = {"temperature": 0}  # 随机采样可能会打断公式标记
        self.client = ollama.Client()
        
    def translate(self, text):
        system_prompt = """You are a professional financial advisor who is also proficient in English and Chinese. You need to translate English to Chinese accurately and professionally by preserving the right financial context. 请将以下英文内容翻译为中文，要求精准且专业，符合中文表述习惯和标点符号的使用规则。不需要翻译数字及常见人名。译文的长度与英文原文相差不大，确保保留原文的专业性和逻辑性。
        示例：
        英文：
        "To achieve long-term financial growth, it's essential to diversify your investment portfolio by allocating assets across different sectors, such as technology, healthcare, and real estate."
        中文：
        "为了实现长期的财务增长，必须通过在科技、医疗和房地产等不同领域分配资产来分散投资组合的风险。"
        Requirements:
        - Output only the translated text
        - Do not include words like "翻译:" or "Translation:" or "译文:" or "中文:" or "Chinese:" or "转换:"
        - Preserve all numerical numbers exactly as they appear
        - Do not add any comments, questions, or explanations
        - Do not respond conversationally
        """
        
        user_prompt = f"Translate: {text}"
        
        response = self.client.chat(model=self.model, 
                                  messages=[
                                      {"role": "system", "content": system_prompt},
                                      {"role": "user", "content": user_prompt}
                                  ])
        
        translated_text = response["message"]["content"].strip()
        
        # Post-process to remove any remaining "翻译:" or similar prefixes
        cleaned_text = re.sub(r'^(翻译：|翻译:|Translation:|译文：|译文:|转换：|转换:)\s*', '', translated_text)
        
        return cleaned_text


class OpenAITranslator(BaseTranslator):
    def __init__(self, service, lang_out, lang_in, model):
        lang_out='zh-CN' if lang_out=='auto' else lang_out
        lang_in='en' if lang_in=='auto' else lang_in
        super().__init__(service, lang_out, lang_in, model)
        self.options = {"temperature": 0}  # 随机采样可能会打断公式标记
        # OPENAI_BASE_URL
        # OPENAI_API_KEY
        self.client = openai.OpenAI()

    def translate(self, text) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            **self.options,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional,authentic machine translation engine.",
                },
                {
                    "role": "user",
                    "content": f"Translate the following markdown source text to {self.lang_out}. Keep the formula notation $v*$ unchanged. Output translation directly without any additional text.\nSource Text: {text}\nTranslated Text:",
                },
            ],
        )
        return response.choices[0].message.content.strip()


class AzureTranslator(BaseTranslator):
    def __init__(self, service, lang_out, lang_in, model):
        lang_out='zh-Hans' if lang_out=='auto' else lang_out
        lang_in='en' if lang_in=='auto' else lang_in
        super().__init__(service, lang_out, lang_in, model)

        try:
            api_key = os.environ["AZURE_APIKEY"]
            endpoint = os.environ["AZURE_ENDPOINT"]
            region = os.environ["AZURE_REGION"]
        except KeyError as e:
            missing_var = e.args[0]
            raise ValueError(f"The environment variable '{missing_var}' is required but not set.") from e

        credential = AzureKeyCredential(api_key)
        self.client = TextTranslationClient(
            endpoint=endpoint, credential=credential, region=region
        )

        # https://github.com/Azure/azure-sdk-for-python/issues/9422
        logger = logging.getLogger("azure.core.pipeline.policies.http_logging_policy")
        logger.setLevel(logging.WARNING)

    def translate(self, text) -> str:
        response = self.client.translate(
            body=[text],
            from_language=self.lang_in,
            to_language=[self.lang_out],
        )

        translated_text = response[0].translations[0].text
        return translated_text
