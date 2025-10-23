#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio Web UI for vLLM Backend (Qwen3-Omni & other multi-modal models)

Features:
- Connects to a remote vLLM OpenAI-compatible API service.
- Supports multi-modal inputs: text, images, videos, and audio.
- Renders uploaded media directly in the chatbot UI for a better user experience.
- Handles multi-turn conversations robustly.
- Designed to be easily adaptable for different multi-modal models like Qwen3-VL and Qwen3-Omni.

Usage:
python web_ui_qwen3_omni.py \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --openai-base-url http://<YOUR_VLLM_SERVER_IP>:8000/v1 \
  --openai-api-key your_api_key \
  --ui-host 0.0.0.0 \
  --ui-port 7860
"""

import os
import base64
import mimetypes
from urllib.parse import quote
from typing import List, Dict, Any, Tuple, Optional, Generator
from argparse import ArgumentParser

import gradio as gr
from openai import OpenAI

# ----------------------------
# CLI Arguments
# ----------------------------

def get_args():
    """Parses command-line arguments."""
    p = ArgumentParser(description="Gradio Web UI for vLLM OpenAI-compatible API")
    
    # vLLM OpenAI-compatible service details
    p.add_argument("--openai-base-url", type=str, default="http://127.0.0.1:8000/v1",
                   help="Base URL for the vLLM OpenAI-compatible service (e.g., http://IP:PORT/v1)")
    p.add_argument("--openai-api-key", type=str, default="EMPTY",
                   help="API key for the vLLM service")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
                   help="Model name to use, as exposed by the vLLM service")

    # Gradio UI settings
    p.add_argument("--ui-host", type=str, default="0.0.0.0",
                   help="Host address for the Gradio UI")
    p.add_argument("--ui-port", type=int, default=7860,
                   help="Port for the Gradio UI")
    p.add_argument("--share", action="store_true",
                   help="Create a public Gradio share link")
    p.add_argument("--public-file-base", type=str, default=None,
                   help="(Optional) Publicly accessible URL prefix for large files like videos. "
                        "Example: http://<UI_HOST>:<UI_PORT>/file=")

    # Model generation parameters
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)

    return p.parse_args()

# ----------------------------
# Helper Functions
# ----------------------------

# Define supported file extensions for different modalities
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".aac", ".m4a"}

def get_file_type(path: str) -> Optional[str]:
    """Determines the modality type of a file based on its extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    return None

def to_data_uri(path: str) -> str:
    """Converts a local file to a data URI."""
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        mime_type = "application/octet-stream"
    
    with open(path, "rb") as f:
        encoded_content = base64.b64encode(f.read()).decode("utf-8")
    
    return f"data:{mime_type};base64,{encoded_content}"

def to_public_url(abs_path: str, public_file_base: str) -> str:
    """Constructs a public URL for a local file, accessible by the vLLM server."""
    return f"{public_file_base}{quote(abs_path)}"

def prepare_api_message_content(
    text: Optional[str],
    files: List[str],
    public_file_base: Optional[str]
) -> List[Dict[str, Any]]:
    """
    Prepares the 'content' list for the OpenAI API message, handling text and multi-modal files.
    - Images are sent as data URIs.
    - Videos and Audios are sent as public URLs if `public_file_base` is provided, otherwise as data URIs.
    """
    content: List[Dict[str, Any]] = []
    
    # Add text part first, if it exists
    if text and text.strip():
        content.append({"type": "text", "text": text.strip()})

    # Add media parts
    for file_path in files:
        abs_path = os.path.abspath(file_path)
        file_type = get_file_type(abs_path)

        if file_type == "image":
            content.append({
                "type": "image_url",
                "image_url": {"url": to_data_uri(abs_path)}
            })
        elif file_type == "video":
            # NOTE: vLLM's OpenAI API extension for Qwen models uses 'input_video'
            url = to_public_url(abs_path, public_file_base) if public_file_base else to_data_uri(abs_path)
            content.append({"type": "input_video", "input_video": {"url": url}})
        elif file_type == "audio":
            # NOTE: Assuming 'input_audio' follows the same pattern as 'input_video' for Qwen-Omni
            url = to_public_url(abs_path, public_file_base) if public_file_base else to_data_uri(abs_path)
            content.append({"type": "input_audio", "input_audio": {"url": url}})
        else:
            print(f"Warning: Unsupported file type skipped: {os.path.basename(abs_path)}")

    return content


# ----------------------------
# Gradio Application Core
# ----------------------------

def build_app(client: OpenAI, args: ArgumentParser) -> gr.Blocks:
    """Builds the Gradio application interface."""

    with gr.Blocks(theme=gr.themes.Soft(), css="""
        .gradio-container {max-width: 90% !important;}
        .message-buttons {gap: 5px; margin-top: 5px; justify-content: flex-end;}
        .message-buttons button {flex-grow: 0; min-width: 80px;}
    """) as demo:
        # --- State Management ---
        # `api_history`: Stores the conversation in OpenAI API format (with data URIs).
        api_history = gr.State(value=[])

        # --- UI Layout ---
        gr.Markdown(
            "<h1 align='center'>Qwen3-Omni Web UI (via vLLM)</h1>"
            "<p align='center'>A general-purpose Gradio UI to interact with multi-modal models served by vLLM.</p>"
        )

        chatbot = gr.Chatbot(
            label="Conversation",
            bubble_full_width=False,
            height=600,
            avatar_images=(None, "https://qianwen-res.oss-cn-beijing.aliyuncs.com/logo_qwen.jpg")
        )

        with gr.Row():
            file_uploader = gr.Files(
                label="Upload Files (Image, Video, Audio)",
                file_count="multiple",
                type="filepath",
                scale=1
            )
            text_input = gr.Textbox(
                label="Type your message...",
                placeholder="Ask a question or describe the uploaded files...",
                lines=4,
                scale=3
            )

        with gr.Row():
            submit_btn = gr.Button("🚀 Send", variant="primary")
            retry_btn = gr.Button("🤔 Regenerate")
            clear_btn = gr.Button("🧹 Clear")
            
        with gr.Accordion("Generation Parameters", open=False):
            gr_max_tokens = gr.Slider(minimum=512, maximum=8192, value=args.max_tokens, step=256, label="Max Tokens")
            gr_temperature = gr.Slider(minimum=0.1, maximum=1.5, value=args.temperature, step=0.1, label="Temperature")
            gr_top_p = gr.Slider(minimum=0.1, maximum=1.0, value=args.top_p, step=0.05, label="Top P")


        # --- Event Handlers ---

        def handle_user_message(
            text: str,
            files: List[str],
            chat_history: List[Tuple],
            api_history_val: List[Dict]
        ) -> Generator[Tuple, None, None]:
            """
            Processes user input, updates UI, and streams the model's response.
            """
            # 1. Immediately update the UI with the user's message and files
            # This solves the problem of not seeing uploaded files.
            user_display_message = ""
            if files:
                # Gradio's Chatbot can display files by passing a tuple of file paths.
                chat_history.append(((tuple(f for f in files),), None))
            if text.strip():
                user_display_message = text.strip()
                chat_history.append((user_display_message, None))
            
            # Add a placeholder for the assistant's response
            chat_history.append((None, ""))
            yield chat_history, gr.update(value=None), gr.update(value="") # Update chat, clear files, clear text

            # 2. Prepare the message for the OpenAI API
            user_api_content = prepare_api_message_content(text, files, args.public_file_base)
            if not user_api_content:
                chat_history[-1] = (None, "Please provide a message or upload a file.")
                yield chat_history, None, ""
                return

            api_history_val.append({"role": "user", "content": user_api_content})

            # 3. Stream the response from the vLLM server
            try:
                stream = client.chat.completions.create(
                    model=args.model,
                    messages=api_history_val,
                    max_tokens=int(gr_max_tokens.value),
                    temperature=gr_temperature.value,
                    top_p=gr_top_p.value,
                    stream=True,
                )

                assistant_response = ""
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if delta:
                        assistant_response += delta
                        chat_history[-1] = (None, assistant_response)
                        yield chat_history, None, ""
            
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                chat_history[-1] = (None, error_message)
                # Remove the failed user turn from API history to allow retry
                api_history_val.pop() 
                yield chat_history, None, ""
                return

            # 4. Finalize the turn
            api_history_val.append({"role": "assistant", "content": assistant_response})
            yield chat_history, None, ""

        def handle_regenerate(
            chat_history: List[Tuple],
            api_history_val: List[Dict]
        ) -> Generator[Tuple, None, None]:
            """
            Regenerates the last assistant response.
            """
            if not api_history_val or len(api_history_val) < 2:
                yield chat_history, api_history_val
                return

            # Remove the last user and assistant messages from API history
            if api_history_val[-1]["role"] == "assistant":
                api_history_val.pop()
                api_history_val.pop()
                
            # Remove the last user message display (if any) and assistant response from chat history
            # This logic assumes user message and assistant response are the last two entries
            last_assistant_msg_idx = -1
            for i in range(len(chat_history) - 1, -1, -1):
                if chat_history[i][1] is not None: # Found assistant message
                    last_assistant_msg_idx = i
                    break
            if last_assistant_msg_idx != -1:
                # Find the start of the last user turn before this assistant message
                last_user_turn_start_idx = last_assistant_msg_idx -1
                while last_user_turn_start_idx >= 0 and chat_history[last_user_turn_start_idx][1] is None:
                    last_user_turn_start_idx -= 1
                
                chat_history = chat_history[:last_user_turn_start_idx + 1]

            # Re-run the generation by submitting the last user message again
            last_user_msg = api_history_val[-1]
            # Convert API content back to simple text/files for re-submission
            # This is a simplified regeneration. For simplicity, we just resend the last turn.
            
            # A simpler way: just regenerate based on the history before the last assistant message
            
            # Start a new streaming generation with the truncated history
            yield from handle_user_message("", [], chat_history, api_history_val[:-1])

        def handle_clear():
            """Clears the chat history and state."""
            return [], [], None, ""

        # --- Event Listeners ---
        
        # Combine text submit and button click
        submit_triggers = [text_input.submit, submit_btn.click]
        for trigger in submit_triggers:
            trigger.cancel()
            trigger(
                fn=handle_user_message,
                inputs=[text_input, file_uploader, chatbot, api_history],
                outputs=[chatbot, file_uploader, text_input],
                show_progress="full"
            ).then(lambda: gr.update(interactive=True), outputs=[submit_btn])
        
        retry_btn.click(
            fn=handle_regenerate,
            inputs=[chatbot, api_history],
            outputs=[chatbot, api_history],
            show_progress="full"
        )
        
        clear_btn.click(
            fn=handle_clear,
            outputs=[chatbot, api_history, file_uploader, text_input]
        )

    return demo

# ----------------------------
# Main Entry Point
# ----------------------------

def main():
    """Main function to initialize and launch the Gradio app."""
    args = get_args()
    
    # Initialize the OpenAI client to connect to the vLLM server
    client = OpenAI(
        api_key=args.openai_api_key,
        base_url=args.openai_base_url
    )
    
    print("Building Gradio App...")
    demo = build_app(client=client, args=args)
    
    print(f"Launching Gradio UI on http://{args.ui_host}:{args.ui_port}")
    demo.queue().launch(
        server_name=args.ui_host,
        server_port=args.ui_port,
        share=args.share,
        inbrowser=False,
    )

if __name__ == "__main__":
    main()