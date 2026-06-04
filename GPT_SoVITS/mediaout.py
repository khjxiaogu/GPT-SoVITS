# mediaout.py
from pydub import AudioSegment
import numpy as np
import io

def numpy_to_audiosegment(audio_tuple: tuple) -> AudioSegment:
    """
    将 Gradio TTS 输出 (sample_rate, audio_array) 转换为 AudioSegment 对象。
    """
    sample_rate, audio_array = audio_tuple

    # 确定声道数与数据类型
    if audio_array.dtype != np.int16:
        audio_array = (audio_array * 32767).astype(np.int16)

    if audio_array.ndim == 1:
        channels = 1
    elif audio_array.ndim == 2:
        # 通常形状为 (声道数, 采样点数) 或 (采样点数, 声道数)
        channels = audio_array.shape[0] if audio_array.shape[0] < audio_array.shape[1] else audio_array.shape[1]
        # 若数据是 (采样点数, 声道数) 需转置为 pydub 期望的 (声道数, 采样点数) 布局
        if audio_array.shape[1] == channels:
            audio_array = audio_array.T
    else:
        raise ValueError("不支持的音频数组维度")

    return AudioSegment(
        data=audio_array.tobytes(),
        sample_width=2,               # int16 = 2 bytes
        frame_rate=sample_rate,
        channels=channels
    )

def concat_with_silence(segments: list, silence_ms: int = 500) -> AudioSegment:
    """
    将多个 AudioSegment 按顺序拼接，每段之间插入指定时长的静音。
    """
    if not segments:
        return AudioSegment.empty()

    silence = AudioSegment.silent(duration=silence_ms)
    combined = segments[0]
    for seg in segments[1:]:
        combined += silence + seg
    return combined

def export_as_mp3_bytes(audio_segment: AudioSegment) -> bytes:
    """将 AudioSegment 导出为 MP3 字节流。"""
    buffer = io.BytesIO()
    audio_segment.export(buffer, format="mp3")
    return buffer.getvalue()

# 保留原函数以兼容其他调用（若无其他调用可移除）
def save_audio_as_mp3(audio_tuple: tuple) -> bytes:
    seg = numpy_to_audiosegment(audio_tuple)
    return export_as_mp3_bytes(seg)