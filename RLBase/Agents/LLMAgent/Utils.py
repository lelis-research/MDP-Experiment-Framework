from openai import OpenAI


class VulcanLLM:
    def __init__(self, model="qwen3-235b", api_key=None):
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://llm.vulcan.alliancecan.ca/api/v1/"
        )

    def __call__(self, prompt, system=None, max_tokens=512, temperature=0.7):
        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content
