# web_ui.py (Final Version)

import gradio as gr
import os
from typing import Generator, Tuple

# 导入本地模块
from vllm_client import VLLMClient
from multimodal_utils import process_files

# --- 配置 ---
VLLM_BASE_URL = "http://172.17.43.70:8000/v1" 
VLLM_API_KEY = "g203"
VLLM_MODEL_NAME = "qwen3-omni"

TEMP_THUMBNAIL_DIR = "temp_thumbs"
if not os.path.exists(TEMP_THUMBNAIL_DIR):
    os.makedirs(TEMP_THUMBNAIL_DIR)
    
vllm_client = VLLMClient(
    base_url=VLLM_BASE_URL,
    api_key=VLLM_API_KEY,
    model_name=VLLM_MODEL_NAME
)

# --- 核心改动 1: 升级 StreamParser ---
# 现在解析器能识别并区分 "thought" 和 "response" 两种内容类型
class StreamParser:
    def __init__(self):
        self.reset()

    def reset(self):
        self.buffer = ""
        self.state = "response"  # 初始状态为处理回复内容
        self.think_tag_open = "<think>"
        self.think_tag_close = "</think>"

    def parse(self, chunk: str) -> Generator[Tuple[str, str], None, None]:
        self.buffer += chunk
        
        while True:
            if self.state == "response":
                open_tag_pos = self.buffer.find(self.think_tag_open)
                if open_tag_pos != -1:
                    yield "response", self.buffer[:open_tag_pos]
                    self.buffer = self.buffer[open_tag_pos + len(self.think_tag_open):]
                    self.state = "thought"
                else:
                    yield "response", self.buffer
                    self.buffer = ""
                    break
            
            elif self.state == "thought":
                close_tag_pos = self.buffer.find(self.think_tag_close)
                if close_tag_pos != -1:
                    yield "thought", self.buffer[:close_tag_pos]
                    self.buffer = self.buffer[close_tag_pos + len(self.think_tag_close):]
                    self.state = "response"
                else:
                    yield "thought", self.buffer
                    self.buffer = ""
                    break

# --- Gradio UI & 逻辑 ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="vLLM Omni Chat") as app:
    api_history_state = gr.State([])

    gr.Markdown("# 🤖 vLLM Qwen3-Omni 多模态聊天界面")

    chatbot = gr.Chatbot(
        [],
        label="Qwen3-Omni",
        type="messages",
        height=600,
        render_markdown=True,
        avatar_images=(("user.png"), ("bot.png"))
    )

    # --- 核心改动 2: 增加用于展示思考过程的UI组件 ---
    with gr.Accordion("🤔 模型思考过程", open=False) as think_accordion:
        thinking_box = gr.Markdown(value="*这里将实时显示模型的思考步骤...*", label="Thinking")

    with gr.Row():
        file_uploader = gr.File(label="上传文件", file_count="multiple", file_types=["image", "video", "audio"])
    with gr.Row():
        text_input = gr.Textbox(show_label=False, placeholder="请输入文本，或上传文件后直接点击发送...", container=False, scale=7)
        submit_btn = gr.Button("发送", variant="primary", scale=1)
    with gr.Accordion("⚙️ 高级设置", open=False):
        base_url_input = gr.Textbox(label="vLLM Base URL", value=VLLM_BASE_URL)
        api_key_input = gr.Textbox(label="API Key", value=VLLM_API_KEY, type="password")
        model_name_input = gr.Textbox(label="Model Name", value=VLLM_MODEL_NAME)
        update_settings_btn = gr.Button("更新设置")

    # (handle_update_settings 和 add_message_to_history 函数保持不变)
    def add_message_to_history(text, files, chatbot_ui, api_history):
        if not text and not files:
            gr.Warning("请输入文本或上传文件！")
            return chatbot_ui, api_history, text, files, gr.update(value="*等待模型响应...*")
        
        file_paths = [f.name for f in files] if files else []
        api_parts, display_html = process_files(file_paths, TEMP_THUMBNAIL_DIR)
        user_display_message = text + f"<hr>{display_html}" if display_html else text
        api_content_list = [{"type": "text", "text": text}] if text else []
        api_content_list.extend(api_parts)
        api_history.append({"role": "user", "content": api_content_list})
        chatbot_ui.append({"role": "user", "content": user_display_message})
        
        # 清空输入框和文件上传器，并重置思考框
        return chatbot_ui, api_history, "", None, "*模型思考过程将显示在这里...*"

    # --- 核心改动 3: 重构 stream_bot_response 函数 ---
    def stream_bot_response(chatbot_ui, api_history):
        if not api_history or api_history[-1]['role'] != 'user':
            return chatbot_ui, api_history, ""

        chatbot_ui.append({"role": "assistant", "content": ""})

        parser = StreamParser()
        thinking_content = ""
        bot_response_clean = ""
        line_buffer = "" # 行缓冲器
        
        raw_stream = vllm_client.generate_stream(api_history)
        
        for raw_chunk in raw_stream:
            for part_type, content in parser.parse(raw_chunk):
                if not content:
                    continue

                if part_type == "thought":
                    thinking_content += content
                    # 使用字典来同时更新多个组件
                    yield {thinking_box: thinking_content}
                
                elif part_type == "response":
                    line_buffer += content
                    bot_response_clean += content
                    
                    # 实现“行缓冲”逻辑
                    if '\n' in line_buffer or len(line_buffer) > 80:
                        chatbot_ui[-1]["content"] = bot_response_clean
                        yield {chatbot: chatbot_ui}
                        line_buffer = "" # 刷新后清空行缓冲

        # 循环结束后，确保所有剩余内容都被刷新到UI
        chatbot_ui[-1]["content"] = bot_response_clean
        api_history.append({"role": "assistant", "content": bot_response_clean})
        # 最后一次 yield 完整的状态
        yield {chatbot: chatbot_ui, api_history_state: api_history, thinking_box: thinking_content}

    # --- 核心改动 4: 更新事件绑定以包含新组件 ---
    submit_btn.click(
        add_message_to_history,
        inputs=[text_input, file_uploader, chatbot, api_history_state],
        outputs=[chatbot, api_history_state, text_input, file_uploader, thinking_box]
    ).then(
        stream_bot_response,
        inputs=[chatbot, api_history_state],
        outputs=[chatbot, api_history_state, thinking_box]
    )
    
    text_input.submit(
        add_message_to_history,
        inputs=[text_input, file_uploader, chatbot, api_history_state],
        outputs=[chatbot, api_history_state, text_input, file_uploader, thinking_box]
    ).then(
        stream_bot_response,
        inputs=[chatbot, api_history_state],
        outputs=[chatbot, api_history_state, thinking_box]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=False, inbrowser=True)