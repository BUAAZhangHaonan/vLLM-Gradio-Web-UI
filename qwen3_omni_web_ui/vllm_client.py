# vllm_client.py (Updated)

from openai import OpenAI
from typing import List, Dict, Any, Generator


class VLLMClient:
    """
    一个用于与 vLLM 的 OpenAI 兼容 API 进行交互的客户端。
    """

    def __init__(self, base_url: str, api_key: str, model_name: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model_name = model_name
        self._initialize_client()

    def _initialize_client(self):
        """初始化 OpenAI 客户端。"""
        try:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        except Exception as e:
            print(f"初始化 OpenAI 客户端失败: {e}")
            self.client = None

    def update_config(self, base_url: str, api_key: str, model_name: str):
        """动态更新客户端配置。"""
        self.base_url = base_url
        self.api_key = api_key
        self.model_name = model_name
        self._initialize_client()
        print("VLLM 客户端配置已更新。")

    def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> Generator[str, None, None]:
        """
        以流式方式调用 vLLM 的 chat completions 接口。
        """
        if not self.client:
            yield "错误：vLLM 客户端未初始化。请检查配置。"
            return

        try:
            # --- 核心改动 ---
            # 增加了 stop 参数，防止模型生成多余的控制字符污染输出流。
            # Qwen 系列模型通常使用 <|im_end|> 作为对话结束符。
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                stream=True,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["<|im_end|>"]
            )
            for chunk in stream:
                content = chunk.choices[0].delta.content
                # 即使 content 是 None 或空字符串，也继续处理，只在有真实内容时 yield
                if content is not None:
                    # 按照您的要求，在终端打印原始流数据以供调试
                    print(content, end='', flush=True)
                    yield content
        except Exception as e:
            error_message = f"与 vLLM 通信时出错: {e}"
            print(error_message)
            yield error_message
