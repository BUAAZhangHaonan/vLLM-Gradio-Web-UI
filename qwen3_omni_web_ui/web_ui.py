# web_ui.py
import gradio as gr
import os
from typing import Generator

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


class StreamParser:
    def __init__(self):
        self.reset()

    def reset(self):
        self.buffer = ""
        self.is_inside_think_tag = False
        self.think_tag_open = "<think>"
        self.think_tag_close = "</think>"

    def parse(self, chunk: str) -> Generator[str, None, None]:
        self.buffer += chunk

        while True:
            if not self.is_inside_think_tag:
                open_tag_pos = self.buffer.find(self.think_tag_open)
                if open_tag_pos != -1:
                    yield self.buffer[:open_tag_pos]
                    self.buffer = self.buffer[open_tag_pos +
                                              len(self.think_tag_open):]
                    self.is_inside_think_tag = True
                else:
                    yield self.buffer
                    self.buffer = ""
                    break

            if self.is_inside_think_tag:
                close_tag_pos = self.buffer.find(self.think_tag_close)
                if close_tag_pos != -1:
                    self.buffer = self.buffer[close_tag_pos +
                                              len(self.think_tag_close):]
                    self.is_inside_think_tag = False
                else:
                    self.buffer = ""
                    break


# --- Gradio UI & 逻辑 ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="vLLM Omni Chat") as app:
    api_history_state = gr.State([])

    gr.Markdown("# 🤖 vLLM Qwen3-Omni 多模态聊天界面")
    gr.Markdown("![Qwen3-Omni](images/qwen3-omni-logo.png)")

    chatbot = gr.Chatbot(
        [],
        label="Qwen3-Omni",
        type="messages",
        height=600,
        render_markdown=True,
        avatar_images=(("images/User.png"), ("images/Qwen.png"))
    )

    with gr.Row():
        file_uploader = gr.File(label="上传文件", file_count="multiple", file_types=[
                                "image", "video", "audio"])
    with gr.Row():
        text_input = gr.Textbox(
            show_label=False, placeholder="请输入文本，或上传文件后直接点击发送...", container=False, scale=7)
        submit_btn = gr.Button("发送", variant="primary", scale=1)
    with gr.Accordion("⚙️ 高级设置", open=False):
        base_url_input = gr.Textbox(label="vLLM Base URL", value=VLLM_BASE_URL)
        api_key_input = gr.Textbox(
            label="API Key", value=VLLM_API_KEY, type="password")
        model_name_input = gr.Textbox(
            label="Model Name", value=VLLM_MODEL_NAME)
        update_settings_btn = gr.Button("更新设置")

    def handle_update_settings(base_url, api_key, model_name):
        vllm_client.update_config(base_url, api_key, model_name)
        gr.Info("设置已更新！")

    def add_message_to_history(text, files, chatbot_ui, api_history):
        if not text and not files:
            gr.Warning("请输入文本或上传文件！")
            return chatbot_ui, api_history, text, files

        file_paths = [f.name for f in files] if files else []
        api_parts, display_html = process_files(file_paths, TEMP_THUMBNAIL_DIR)

        user_display_message = text if text else ""
        if display_html:
            user_display_message += f"<hr>{display_html}"

        api_content_list = []
        if text:
            api_content_list.append({"type": "text", "text": text})
        api_content_list.extend(api_parts)

        api_history.append({"role": "user", "content": api_content_list})
        chatbot_ui.append({"role": "user", "content": user_display_message})

        return chatbot_ui, api_history, "", None

    def stream_bot_response(chatbot_ui, api_history):
        if not api_history or api_history[-1]['role'] != 'user':
            return chatbot_ui, api_history

        chatbot_ui.append({"role": "assistant", "content": ""})
        parser = StreamParser()
        bot_response_clean = ""
        bot_response_display = ""  # 用于前端，含 <br>

        raw_stream = vllm_client.generate_stream(api_history)
        for raw_chunk in raw_stream:
            for clean_chunk in parser.parse(raw_chunk):
                if clean_chunk:
                    bot_response_clean += clean_chunk
                    # 替换 \n 为 <br>，这是行业标准做法
                    display_chunk = clean_chunk.replace("\n", "<br>")
                    bot_response_display += display_chunk
                    chatbot_ui[-1]["content"] = bot_response_display
                    yield chatbot_ui, api_history

        api_history.append(
            {"role": "assistant", "content": bot_response_clean})
        yield chatbot_ui, api_history

    # --- 事件绑定部分保持不变 ---
    submit_btn.click(
        add_message_to_history,
        inputs=[text_input, file_uploader, chatbot, api_history_state],
        outputs=[chatbot, api_history_state, text_input, file_uploader]
    ).then(
        stream_bot_response,
        inputs=[chatbot, api_history_state],
        outputs=[chatbot, api_history_state]
    )

    text_input.submit(
        add_message_to_history,
        inputs=[text_input, file_uploader, chatbot, api_history_state],
        outputs=[chatbot, api_history_state, text_input, file_uploader]
    ).then(
        stream_bot_response,
        inputs=[chatbot, api_history_state],
        outputs=[chatbot, api_history_state]
    )

    update_settings_btn.click(
        handle_update_settings,
        inputs=[base_url_input, api_key_input, model_name_input],
        outputs=[]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=False, inbrowser=True)
