#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3 MultiModal 前端可视化 (连接 vLLM OpenAI 兼容服务)
- 支持纯文本、视觉语言、全模态模型
- 支持图片、视频、音频上传和显示
- 支持局域网访问
- 支持多轮对话

python web_ui_qwen3_omni.py \
  --openai-base-url http://172.17.43.70:8000/v1 \
  --openai-api-key g203 \
  --model qwen3-omni \
  --ui-host 0.0.0.0 \
  --ui-port 7860
"""

from typing import List, Dict, Any, Optional, Union

import gradio as gr
from openai import OpenAI
from argparse import ArgumentParser

from config import UI_CONFIG, DEFAULT_PARAMS, MODEL_CONFIGS
from multimodal_utils import make_user_content, detect_model_type


# ----------------------------
# CLI 参数
# ----------------------------

def get_args():
    parser = ArgumentParser(
        description="Qwen3 MultiModal UI - Connect to vLLM OpenAI Service")
    # vLLM OpenAI 兼容服务地址与鉴权
    parser.add_argument("--openai-base-url", type=str, default="http://172.17.43.70:8000/v1",
                        help="vLLM 服务的 OpenAI 兼容 base_url, 例如 http://IP:PORT/v1")
    parser.add_argument("--openai-api-key", type=str, default="g203",
                        help="vLLM 服务设置的 --api-key")
    parser.add_argument("--model", type=str, default="qwen3-omni",
                        help="服务端暴露的模型名")

    # 前端 UI 暴露
    parser.add_argument("--ui-host", type=str, default="0.0.0.0",
                        help="Gradio 监听地址, 0.0.0.0 便于局域网访问")
    parser.add_argument("--ui-port", type=int, default=7860, help="Gradio 端口")
    parser.add_argument("--share", action="store_true",
                        help="Gradio share(一般内网不需要)")

    # 视频/音频直链可选: 让 vLLM 服务端能拉到你上传的文件
    parser.add_argument("--public-file-base", type=str, default=None,
                        help="(可选)将本地上传文件映射为服务端可访问的 URL 前缀, 例如 http://IP:PORT/file=")

    # 采样与限长
    parser.add_argument("--max-tokens", type=int,
                        default=DEFAULT_PARAMS["max_tokens"])
    parser.add_argument("--temperature", type=float,
                        default=DEFAULT_PARAMS["temperature"])
    parser.add_argument("--top_p", type=float, default=DEFAULT_PARAMS["top_p"])

    return parser.parse_args()


# ----------------------------
# 工具函数
# ----------------------------

def test_connection(base_url: str, api_key: str) -> bool:
    """测试与vLLM服务的连接"""
    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        # 尝试获取模型列表
        response = client.models.list()
        print(f"✅ 连接成功，可用模型: {[model.id for model in response.data]}")
        return True
    except Exception as e:
        print(f"❌ 连接失败: {str(e)}")
        return False


def get_available_models(client: OpenAI) -> List[str]:
    """获取可用的模型列表"""
    try:
        response = client.models.list()
        return [model.id for model in response.data]
    except Exception as e:
        print(f"获取模型列表失败: {str(e)}")
        return []


# 新增：统一打包 Gradio submit 的输出（与 submit/query 绑定的 outputs 顺序一致）
def _pack_submit_outputs(chat, backend_history, pending_files, textbox_update, audio_value, connection_ok):
    # 对应 outputs: [chatbot, state_backend_history, state_pending_files, query, audio_output, state_connection_ok]
    return (chat, backend_history, pending_files, textbox_update, audio_value, connection_ok)


# 新增：统一打包 Gradio regen 的输出（与 regen_btn 绑定的 outputs 顺序一致）
def _pack_regen_outputs(chat, audio_value, connection_ok):
    # 对应 outputs: [chatbot, audio_output, state_connection_ok]
    return (chat, audio_value, connection_ok)


# 新增：将内部的 dict 列表转换为 Gradio 默认 Chatbot 接受的 (user, assistant) 对列表
def _convert_chat_to_pairs(chat_list):
    """
    chat_list 示例: [
      {"role": "user", "content": "..."},
      {"role": "assistant", "content": "..."},
      ...
    ]
    转换为: [ (user1, assistant1), (user2, assistant2), ... ]
    如果最后只有 user 没有 assistant，则 assistant 置为 ""，保证前端能正确显示未完的消息。
    """
    if not chat_list:
        return []
    pairs = []
    last_user = None
    for msg in chat_list:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            # 覆盖上一次未配对的 user（保留最新用户输入）
            last_user = content
        elif role == "assistant":
            # 如果没有对应的 user，使用空字符串占位
            pairs.append((last_user if last_user is not None else "", content))
            last_user = None
    # 如果以 user 结尾，没有 assistant，则加入一条空助手占位
    if last_user is not None:
        pairs.append((last_user, ""))
    return pairs


# 修改打包函数以在返回前转换 chat 格式
def _pack_submit_outputs(chat, backend_history, pending_files, textbox_update, audio_value, connection_ok):
    # 对应 outputs: [chatbot, state_backend_history, state_pending_files, query, audio_output, state_connection_ok]
    formatted_chat = _convert_chat_to_pairs(chat)
    return (formatted_chat, backend_history, pending_files, textbox_update, audio_value, connection_ok)


def _pack_regen_outputs(chat, audio_value, connection_ok):
    # 对应 outputs: [chatbot, audio_output, state_connection_ok]
    formatted_chat = _convert_chat_to_pairs(chat)
    return (formatted_chat, audio_value, connection_ok)

# ----------------------------
# 核心: 创建 Gradio App
# ----------------------------


def build_app(client: OpenAI, model_name: str, max_tokens: int, temperature: float, top_p: float,
              public_file_base: Optional[str], connection_ok: bool = False, available_models: List[str] = None) -> gr.Blocks:
    """
    创建多模态聊天界面
    """
    # 检测模型类型
    detected_model_type = detect_model_type(model_name)
    model_config = MODEL_CONFIGS.get(
        detected_model_type, MODEL_CONFIGS["qwen3-omni"])

    # 获取可用模型列表
    if available_models is None:
        available_models = get_available_models(client)

    with gr.Blocks(
        title=UI_CONFIG["title"],
        theme=gr.themes.Soft(),
        css="""
        .chat-container { max-height: 600px; overflow-y: auto; }
        .media-preview { max-width: 100%; max-height: 200px; }
        .gradio-container { max-width: 1200px; margin: 0 auto; }
        .chat-message { margin-bottom: 10px; }
        .user-message { text-align: right; color: #333; }
        .assistant-message { text-align: left; color: #333; }
        img.media-preview { border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .connection-status { padding: 10px; margin: 10px 0; border-radius: 5px; }
        .connection-status.connected { background-color: #d4edda; color: #155724; }
        .connection-status.disconnected { background-color: #f8d7da; color: #721c24; }
        """
    ) as demo:
        gr.Markdown(
            "<p align='center'><img src='https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3-VL/qwen3vllogo.png' "
            "style='height: 72px'/></p>"
        )
        gr.Markdown(
            f"<center><h1>Qwen3 MultiModal ({detected_model_type.upper()})</h1></center>")

        # 连接状态显示 - 根据实际连接状态初始化
        if connection_ok:
            status_html = f"""
            <div class='connection-status connected'>
            ✅ 连接成功<br>
            模型: {model_name}<br>
            可用模型: {', '.join(available_models)}
            </div>
            """
        else:
            status_html = "<div class='connection-status disconnected'>❌ 连接失败</div>"

        connection_status = gr.HTML(
            value=status_html,
            label="连接状态"
        )

        # 模型信息
        gr.Markdown(
            f"<center>模型类型: {model_config['description']} | 连接至: {client.base_url}</center>")

        # 状态管理 - 使用默认类型以配合我们传入的 (user, assistant) 对列表
        chatbot = gr.Chatbot(
            label=f"Qwen3-{detected_model_type.upper()}",
            height=UI_CONFIG["height"],
            avatar_images=(
                "https://cdn-icons-png.flaticon.com/512/4712/4712035.png",  # user
                "https://cdn-icons-png.flaticon.com/512/4712/4712139.png"   # assistant
            )
        )

        # 输入组件
        with gr.Row():
            query = gr.Textbox(
                lines=2,
                label="输入文本",
                placeholder="输入您的消息...",
                container=True
            )

        with gr.Row():
            # 根据模型支持的模态类型显示上传按钮
            upload_components = []
            if "image" in model_config["supports"]:
                img_upload = gr.UploadButton(
                    "🖼️ 上传图片",
                    file_types=["image"],
                    visible=True
                )
                upload_components.append(img_upload)
            else:
                img_upload = gr.UploadButton(visible=False)

            if "video" in model_config["supports"]:
                video_upload = gr.UploadButton(
                    "🎬 上传视频",
                    file_types=["video"],
                    visible=True
                )
                upload_components.append(video_upload)
            else:
                video_upload = gr.UploadButton(visible=False)

            if "audio" in model_config["supports"]:
                audio_upload = gr.UploadButton(
                    "🎵 上传音频",
                    file_types=["audio"],
                    visible=True
                )
                upload_components.append(audio_upload)
            else:
                audio_upload = gr.UploadButton(visible=False)

            submit_btn = gr.Button("🚀 发送")
            regen_btn = gr.Button("🤔 重试")
            clear_btn = gr.Button("🧹 清空")

        # 隐藏的音频播放器（用于播放模型生成的音频）
        audio_output = gr.Audio(visible=False, autoplay=True)

        # 状态变量
        state_backend_history = gr.State(value=[])  # 后端消息历史 (用于OpenAI API)
        state_pending_files = gr.State(value=[])   # 待发送文件
        state_connection_ok = gr.State(value=connection_ok)  # 连接状态

        # --- 事件函数 ---

        def on_upload(pending_files: List[str], uploaded_file) -> List[str]:
            """处理文件上传"""
            if uploaded_file is not None:
                files = list(pending_files or [])
                files.append(uploaded_file.name)
                return files
            return pending_files

        def format_message_for_display(role: str, content: Union[str, List[Dict[str, Any]]]) -> str:
            """格式化消息用于显示，支持图片HTML渲染"""
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                display_parts = []

                for item in content:
                    if item["type"] == "text":
                        display_parts.append(item["text"])
                    elif item["type"] in ["image_url", "image"] and "url" in item.get("image_url", {}):
                        # 直接显示图片URL，Gradio会自动渲染为图片
                        img_url = item["image_url"]["url"]
                        if img_url.startswith("data:"):
                            # 内联图片
                            display_parts.append(
                                f'<img src="{img_url}" class="media-preview" alt="用户图片">')
                        else:
                            # 外部URL图片
                            display_parts.append(
                                f'<img src="{img_url}" class="media-preview" alt="用户图片">')
                    elif item["type"] in ["input_video", "video"] and "url" in item.get("input_video", {}):
                        # 视频显示为文本链接
                        video_url = item["input_video"]["url"]
                        display_parts.append(
                            f'[视频] <a href="{video_url}" target="_blank">查看视频</a>')
                    elif item["type"] in ["input_audio", "audio"] and "url" in item.get("input_audio", {}):
                        # 音频显示为文本链接
                        audio_url = item["input_audio"]["url"]
                        display_parts.append(
                            f'[音频] <a href="{audio_url}" target="_blank">播放音频</a>')

                return "<br>".join(display_parts)
            else:
                return str(content)

        def on_submit(chat: List[Dict[str, Any]],
                      backend_history: List[Dict[str, Any]],
                      pending_files: List[str],
                      text: str,
                      connection_ok: bool):
            """处理用户提交"""
            if not connection_ok:
                error_msg = "❌ 连接未建立或已断开，请检查vLLM服务是否正常运行"
                chat = list(chat or [])
                chat.append({"role": "assistant", "content": error_msg})
                # 使用打包函数，确保返回 6 个输出
                yield _pack_submit_outputs(chat, backend_history, pending_files, gr.update(value=""), None, connection_ok)
                return

            print(f"收到消息: {text}")  # 简化调试日志

            # 组装用户消息内容 (用于发送给OpenAI API)
            user_content = make_user_content(
                text, pending_files or [], public_file_base, detected_model_type
            )

            if not user_content and not pending_files and not text:
                # 空输入则无动作：使用打包函数，不返回额外 HTML
                yield _pack_submit_outputs(chat, backend_history, pending_files, gr.update(value=""), None, connection_ok)
                return

            # 更新后端历史 (用于OpenAI API)
            backend_history = list(backend_history or [])
            backend_history.append({"role": "user", "content": user_content})

            # 更新前端聊天显示
            chat = list(chat or [])

            # 格式化用户消息用于显示
            user_display_content = format_message_for_display(
                "user", user_content)
            chat.append({"role": "user", "content": user_display_content})

            # 流式请求模型
            try:
                print("开始流式请求...")  # 简化调试日志
                stream = client.chat.completions.create(
                    model=model_name,  # 使用传入的模型名，而不是硬编码
                    messages=backend_history,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True,
                )

                partial_response = ""
                first_chunk = True

                for chunk in stream:
                    # 检查 chunk.choices 是否存在以及是否非空
                    if not chunk.choices:
                        continue

                    # 检查 choices[0].delta 是否存在以及是否有 content
                    delta = chunk.choices[0].delta
                    if delta is None:
                        continue

                    content = delta.content
                    if content is None:
                        continue

                    # 累积内容
                    partial_response += content

                    # 更新助手消息显示
                    if first_chunk:
                        # 如果最后一条消息是用户消息，则添加新的助手消息
                        if chat and chat[-1]["role"] == "user":
                            chat.append(
                                {"role": "assistant", "content": partial_response})
                        first_chunk = False
                    else:
                        # 否则更新最后一条助手消息
                        chat[-1]["content"] = partial_response

                    # 每次 yield 都使用打包函数（严格 6 项）
                    yield _pack_submit_outputs(chat, backend_history, pending_files, gr.update(value=""), None, connection_ok)

            except Exception as e:
                error_msg = f"模型调用出错: {str(e)}"
                print(f"错误: {error_msg}")  # 简化错误日志

                # 更新连接状态
                connection_ok = False

                if chat and chat[-1]["role"] == "user":
                    chat.append({"role": "assistant", "content": error_msg})
                else:
                    chat[-1]["content"] = error_msg

                # 使用打包函数返回（移除额外的 status_html）
                yield _pack_submit_outputs(chat, backend_history, pending_files, gr.update(value=""), None, connection_ok)
                return

            print("流式请求完成")  # 简化调试日志

            # 将助手响应添加到后端历史
            backend_history.append({
                "role": "assistant",
                "content": [{"type": "text", "text": partial_response}]
            })

            # 清空待发送文件
            pending_files = []

            # 最后 yield 一次，确保状态更新（保持 6 项）
            yield _pack_submit_outputs(chat, backend_history, pending_files, gr.update(value=""), None, connection_ok)

        def on_regen(chat: List[Dict[str, Any]],
                     backend_history: List[Dict[str, Any]],
                     connection_ok: bool):
            """重试功能"""
            if not connection_ok:
                error_msg = "❌ 连接未建立或已断开，请检查vLLM服务是否正常运行"
                chat = list(chat or [])
                chat.append({"role": "assistant", "content": error_msg})
                # regen 绑定 outputs 为 [chatbot, audio_output, state_connection_ok] -> 返回 3 项
                yield _pack_regen_outputs(chat, None, connection_ok)
                return

            if not backend_history:
                print("Debug: No history to regenerate")  # 调试日志
                return chat, None, connection_ok

            # 移除最后一个assistant消息
            if backend_history and backend_history[-1]["role"] == "assistant":
                backend_history = backend_history[:-1]

            # 重新生成最后一条消息
            if backend_history and backend_history[-1]["role"] == "user":
                # 获取最后一条用户消息用于显示
                last_user_msg = backend_history[-1]
                user_content = last_user_msg.get("content", "")
                if isinstance(user_content, list):
                    # 如果是多模态内容，提取文本部分
                    text_parts = []
                    for item in user_content:
                        if item.get("type") == "text":
                            text_parts.append(item["text"])
                    user_display = " ".join(text_parts)
                else:
                    user_display = str(user_content)

                # 添加用户消息到前端聊天记录
                if chat and chat[-1]["role"] == "assistant":
                    # 保留最后的助手消息，添加用户消息
                    pass
                else:
                    chat.append({"role": "user", "content": user_display})

                try:
                    stream = client.chat.completions.create(
                        model=model_name,  # 使用传入的模型名
                        messages=backend_history,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        stream=True,
                    )

                    partial_response = ""
                    first_chunk = True

                    for chunk in stream:
                        delta = chunk.choices[0].delta.content or ""
                        if delta:
                            partial_response += delta
                            if first_chunk:
                                if chat and chat[-1]["role"] == "user":
                                    # 添加助手消息
                                    chat.append(
                                        {"role": "assistant", "content": partial_response})
                                first_chunk = False
                            else:
                                # 更新最后一条助手消息
                                chat[-1]["content"] = partial_response
                            # regen 也要返回与绑定一致的 3 项
                            yield _pack_regen_outputs(chat, None, connection_ok)

                except Exception as e:
                    error_msg = f"重试出错: {str(e)}"
                    print(f"Error: {error_msg}")  # 错误日志

                    # 更新连接状态
                    connection_ok = False

                    if chat and chat[-1]["role"] == "user":
                        chat.append(
                            {"role": "assistant", "content": error_msg})
                    else:
                        chat[-1]["content"] = error_msg
                    # 移除多余的 status_html，仅返回绑定的 3 项
                    yield _pack_regen_outputs(chat, None, connection_ok)
                    return

                # 更新后端历史
                backend_history.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": partial_response}]
                })

            # 最终返回与 regen 绑定一致的 3 项
            yield _pack_regen_outputs(chat, None, connection_ok)

        def on_clear():
            """清空对话"""
            return [], [], [], gr.HTML(value="<div class='connection-status connected'>✅ 连接正常</div>")

        def on_test_connection(base_url: str, api_key: str, model_name: str):
            """测试连接"""
            try:
                client = OpenAI(api_key=api_key, base_url=base_url)
                if test_connection(base_url, api_key):
                    # 尝试获取模型信息
                    available_models = get_available_models(client)
                    if model_name in available_models:
                        status_html = f"""
                        <div class='connection-status connected'>
                        ✅ 连接成功<br>
                        模型: {model_name}<br>
                        可用模型: {', '.join(available_models)}
                        </div>
                        """
                        return status_html, True, available_models
                    else:
                        status_html = f"""
                        <div class='connection-status disconnected'>
                        ⚠️ 连接成功但模型不存在<br>
                        请求的模型: {model_name}<br>
                        可用模型: {', '.join(available_models)}
                        </div>
                        """
                        return status_html, False, available_models
                else:
                    status_html = "<div class='connection-status disconnected'>❌ 连接失败</div>"
                    return status_html, False, []
            except Exception as e:
                status_html = f"<div class='connection-status disconnected'>❌ 连接失败: {str(e)}</div>"
                return status_html, False, []

        # --- 事件绑定 ---

        # 文件上传事件
        if "image" in model_config["supports"]:
            img_upload.upload(
                on_upload,
                [state_pending_files, img_upload],
                [state_pending_files],
                show_progress=True
            )

        if "video" in model_config["supports"]:
            video_upload.upload(
                on_upload,
                [state_pending_files, video_upload],
                [state_pending_files],
                show_progress=True
            )

        if "audio" in model_config["supports"]:
            audio_upload.upload(
                on_upload,
                [state_pending_files, audio_upload],
                [state_pending_files],
                show_progress=True
            )

        # 提交事件
        submit_event = submit_btn.click(
            on_submit,
            [chatbot, state_backend_history,
                state_pending_files, query, state_connection_ok],
            [chatbot, state_backend_history, state_pending_files,
                query, audio_output, state_connection_ok],
            show_progress=True
        )

        # 回车提交
        query.submit(
            on_submit,
            [chatbot, state_backend_history,
                state_pending_files, query, state_connection_ok],
            [chatbot, state_backend_history, state_pending_files,
                query, audio_output, state_connection_ok],
            show_progress=True
        )

        # 重试事件
        regen_btn.click(
            on_regen,
            [chatbot, state_backend_history, state_connection_ok],
            [chatbot, audio_output, state_connection_ok],
            show_progress=True
        )

        # 清空事件
        clear_btn.click(
            on_clear,
            outputs=[chatbot, state_backend_history,
                     state_pending_files, connection_status]
        )

        # 添加使用说明
        gr.Markdown(
            f"""
            <small>
            <strong>使用说明:</strong><br>
            - 模型支持: {', '.join(model_config['supports'])}<br>
            - 图片会直接显示在聊天中<br>
            - 视频和音频显示为链接形式<br>
            - 多轮对话支持同时上传多种模态文件<br>
            - 如果连接失败，请检查vLLM服务是否正常运行
            </small>
            """
        )

    return demo


# ----------------------------
# 入口
# ----------------------------

def main():
    args = get_args()

    print("="*60)
    print("Qwen3 MultiModal UI 启动中...")
    print(f"OpenAI Base URL: {args.openai_base_url}")
    print(f"API Key: {args.openai_api_key}")
    print(f"Model: {args.model}")
    print(f"UI Host: {args.ui_host}")
    print(f"UI Port: {args.ui_port}")
    print("="*60)

    # 测试连接
    print("正在测试连接...")
    connection_ok = test_connection(args.openai_base_url, args.openai_api_key)

    if not connection_ok:
        print("❌ 连接失败，请检查vLLM服务是否正常运行")
        print("常见问题:")
        print("1. vLLM服务未启动")
        print("2. IP地址或端口错误")
        print("3. API Key不正确")
        print("4. 防火墙阻止连接")
        return

    # 创建OpenAI客户端
    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url)

    # 获取可用模型
    available_models = get_available_models(client)
    print(f"可用模型: {available_models}")

    if args.model not in available_models:
        print(f"⚠️ 请求的模型 '{args.model}' 不在可用模型列表中")
        print(f"可用模型: {available_models}")
        if available_models:
            # 使用第一个可用模型
            args.model = available_models[0]
            print(f"将使用模型: {args.model}")

    # 构建应用
    demo = build_app(
        client=client,
        model_name=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        public_file_base=args.public_file_base,
        connection_ok=connection_ok,
        available_models=available_models,
    )

    # 启动应用
    demo.queue(
        default_concurrency_limit=10,
        max_size=100,
        api_open=True
    ).launch(
        server_name=args.ui_host,
        server_port=args.ui_port,
        share=args.share,
        inbrowser=False,
        show_error=True,
        debug=True
    )


if __name__ == "__main__":
    main()
