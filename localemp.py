# server.py
import asyncio
import websockets
import json
import os
import traceback
import requests
import numpy as np
import inference_webui as inf
import mediaout
sovits = ""
gpt = ""

# 全局消息队列，所有连接收到的有效消息都会放入此队列排队处理
message_queue = asyncio.Queue()

# 全局模型状态（可根据需要改为按配置独立，此处保持全局单一模型）
sovits = ""
gpt = ""

# 内置 ASR（FunASR）模型，懒加载；加载失败只跳过字数校验，不影响正常生成
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_ASR_MODEL_DIR = os.path.join(_PROJECT_ROOT, "tools", "asr", "models")
_ASR_DEVICE = os.environ.get("LOCALEMP_ASR_DEVICE", "cpu")
_asr_model = None
_asr_failed = False


def get_asr_model():
    """加载项目内置的 FunASR 中文模型，失败返回 None。"""
    global _asr_model, _asr_failed
    if _asr_model is not None:
        return _asr_model
    if _asr_failed:
        return None

    try:
        from funasr import AutoModel

        _asr_model = AutoModel(
            model=os.path.join(
                _ASR_MODEL_DIR,
                "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            ),
            vad_model=os.path.join(
                _ASR_MODEL_DIR, "speech_fsmn_vad_zh-cn-16k-common-pytorch"
            ),
            punc_model=os.path.join(
                _ASR_MODEL_DIR, "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
            ),
            device=_ASR_DEVICE,
        )
        print(f"ASR 模型加载完成（device={_ASR_DEVICE}）")
    except Exception as e:
        _asr_failed = True
        print(f"ASR 模型加载失败，跳过字数校验: {e}")
        traceback.print_exc()

    return _asr_model


def recognize_audio(segment):
    """识别一段音频的文本，供 mediaout 的字数校验使用。"""
    model = get_asr_model()
    if model is None:
        return None

    audio = segment.set_frame_rate(16000).set_channels(1)
    samples = np.asarray(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    result = model.generate(input=samples)
    if not result:
        return ""
    return result[0].get("text", "")

def process_data(data):
    """
    处理语音生成请求，返回 MP3 字节数据。
    :param data: 请求 JSON 数据
    :return: 生成的 MP3 字节数据，若失败返回 None
    """
    chara = data.get("chara")
    reqid = data.get("reqid")
    content = data.get("content")
    if not chara or not content or not isinstance(content, list):
        print(f"{reqid}:请求缺少必要字段 (chara/content)，或 content 不是数组")
        return None

    base_dir = os.path.join("kh_models", chara)
    meta_path = os.path.join(base_dir, "meta.json")

    if not os.path.isfile(meta_path):
        print(f"{reqid}:meta.json 不存在: {meta_path}")
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta_content = json.load(f)
    except Exception as e:
        print(f"{reqid}:读取 meta.json 失败: {e}")
        return None

    # 切换模型权重（如需）
    global sovits, gpt
    try:
        if meta_content.get("sovits") != sovits:
            list(inf.change_sovits_weights(meta_content["sovits"], "中文", "中文"))
            sovits = meta_content["sovits"]
        if meta_content.get("gpt") != gpt:
            inf.change_gpt_weights(meta_content["gpt"])
            gpt = meta_content["gpt"]
    except Exception as e:
        print(f"切换模型权重失败: {e}")
        traceback.print_exc()
        # 可根据需要决定是否继续处理

    # ASR 用于每句生成后核对字数；模型加载失败则跳过校验，正常生成
    recognize_fn = recognize_audio if get_asr_model() is not None else None

    generated = []  # [(synthesize, [AudioPiece, ...]), ...]，按内容段顺序保存
    for idx, item in enumerate(content):
        emotion = item.get("emotion")
        text = item.get("text")
        if not emotion or not text:
            print(f"{reqid}:跳过第 {idx} 项：缺少 emotion 或 text")
            continue

        emote_path = os.path.join(base_dir, f"【{emotion}】.mp3")
        if not os.path.isfile(emote_path):
            print(f"{reqid}:参考音频不存在: {emote_path}，跳过该项")
            continue

        def synthesize(sentence, _emote_path=emote_path):
            """单句合成：关闭 TTS 内部切句，切句与空白处理统一交给 mediaout。"""
            tts_gen = inf.get_tts_wav(
                top_k=30,
                ref_wav_path=_emote_path,
                prompt_text=None,
                prompt_language="中文",
                text_language="中文",
                text=sentence,
                how_to_cut="不切"
            )
            return next(tts_gen)

        try:
            print(f"{reqid}:生成语音片段 [{idx}]: emotion={emotion}, text={text}")
            pieces = mediaout.generate_audio_segments(text, synthesize, recognize_fn)
            if pieces:
                generated.append((synthesize, pieces))
        except Exception as e:
            print(f"{reqid}:第 {idx} 项生成失败: {e}")
            traceback.print_exc()

    if not generated:
        print(f"{reqid}:没有成功生成任何音频片段")
        return None

    # 把所有内容段的音频响度统一拉到平均响度
    all_pieces = [piece for _, pieces in generated for piece in pieces]
    mediaout.normalize_loudness(all_pieces)

    # 同一段内容内部的多句用 300ms 间隔；不同内容段之间用 500ms
    audio_segments = [
        mediaout.concat_with_silence(
            [piece.segment for piece in pieces], silence_ms=300
        )
        for _, pieces in generated
    ]
    combined = mediaout.concat_with_silence(audio_segments, silence_ms=500)
    mp3_bytes = mediaout.export_as_mp3_bytes(combined)

    # 发送响应（此处仍由调用方发送，或者在此函数内发送，选择在外部发送以便统一处理异常）
    return mp3_bytes

async def message_processor():
    """
    从消息队列中取出消息并处理，确保串行处理。
    """
    while True:
        item = await message_queue.get()
        if item is None:
            break  # 停止信号
        data, response_url, response_token, reqid = item
        print(f"处理消息 reqid={reqid}，目标 URL={response_url}")
        try:
            mp3_bytes = process_data(data)
            if mp3_bytes is None:
                print(f"reqid={reqid} 处理失败，未生成音频")
                continue
            # 发送响应
            response = requests.post(
                f"{response_url}/aichat/kh$localModelDeployData?reqid={reqid}&type=response",
                data=mp3_bytes,
                headers={"authorization": f"Bearer {response_token}"}
            )
            print(f"发送结束，reqid={reqid}，状态码: {response.status_code}")
        except Exception as e:
            print(f"reqid={reqid} 发送响应失败: {e}")
        finally:
            message_queue.task_done()

async def websocket_listener(config):
    """
    单个 WebSocket 连接监听器，负责连接、重连，并将收到的有效消息放入全局队列。
    :param config: 字典，包含 url 和 token
    """
    url = config["url"]
    token = config["token"]
    reconnect_delay = 1

    while True:
        try:
            print(f"[{url}] 正在连接...")
            async with websockets.connect(
                url + "/aichat/kh$localModelDeploy?model_type=audio",
                extra_headers={"authorization": f"Bearer {token}"}
            ) as websocket:
                print(f"[{url}] 连接成功，等待消息...")
                async for message in websocket:
                    print(f"[{url}] 收到消息...")
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        print(f"[{url}] 收到非 JSON 格式消息，已忽略")
                        continue

                    if data.get("type") != "audio":
                        continue

                    reqid = data.get("reqid")
                    if not reqid:
                        print(f"[{url}] 消息缺少 reqid，忽略")
                        continue

                    # 构造用于发送响应的完整 URL 前缀（http 形式）
                    # 假设 config 中的 url 是 "ws://host:port" 格式，替换为 http
                    response_url = url.replace("ws://", "http://").replace("wss://", "https://")
                    # 将消息放入队列，携带必要的发送信息
                    await message_queue.put((data, response_url, token, reqid))
        except (websockets.exceptions.ConnectionClosed, ConnectionError, OSError) as e:
            print(f"[{url}] 连接断开: {e}，{reconnect_delay}秒后重试...")
        except KeyboardInterrupt:
            print(f"[{url}] 程序被用户中断")
            break
        except Exception as e:
            print(f"[{url}] 未预期错误: {e}，{reconnect_delay}秒后重试...")
        await asyncio.sleep(reconnect_delay)

async def main():
    # 启动消息处理器
    processor_task = asyncio.create_task(message_processor())
    CONFIGS = []
    with open("empconfig.json", "r", encoding="utf-8") as f:
        CONFIGS = json.load(f)
    # 为每个配置启动独立的监听任务
    listener_tasks = [
        asyncio.create_task(websocket_listener(cfg))
        for cfg in CONFIGS
    ]

    try:
        # 等待所有任务（实际上它们会一直运行）
        await asyncio.gather(*listener_tasks)
    except KeyboardInterrupt:
        print("正在停止所有任务...")
    finally:
        # 发送停止信号给处理器
        await message_queue.put(None)
        await processor_task
        for task in listener_tasks:
            task.cancel()
        await asyncio.gather(*listener_tasks, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())