# asr_utils.py
"""内置 FunASR 模型加载与单段识别（懒加载）。

localemp 与 WebUI 的“整段语音”页签共用，避免重复加载模型。
"""

import os
import traceback

import numpy as np

# 项目内相对目录（以项目根目录为工作目录），方便整体迁移
ASR_MODEL_DIR = os.path.join("tools", "asr", "models")
ASR_DEVICE = os.environ.get("LOCALEMP_ASR_DEVICE", "cpu")

_model = None
_load_failed = False


def get_asr_model():
    """加载项目内置的 FunASR 中文模型，失败返回 None。"""
    global _model, _load_failed
    if _model is not None:
        return _model
    if _load_failed:
        return None

    try:
        from funasr import AutoModel

        _model = AutoModel(
            model=os.path.join(
                ASR_MODEL_DIR,
                "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            ),
            vad_model=os.path.join(
                ASR_MODEL_DIR, "speech_fsmn_vad_zh-cn-16k-common-pytorch"
            ),
            punc_model=os.path.join(
                ASR_MODEL_DIR, "punc_ct-transformer_zh-cn-common-vocab272727-pytorch"
            ),
            device=ASR_DEVICE,
        )
        print(f"ASR 模型加载完成（device={ASR_DEVICE}）")
    except Exception as e:
        _load_failed = True
        print(f"ASR 模型加载失败，跳过字数校验: {e}")
        traceback.print_exc()

    return _model


def recognize_audio(segment):
    """识别一段 pydub AudioSegment，返回文本；模型不可用时返回 None。"""
    model = get_asr_model()
    if model is None:
        return None

    audio = segment.set_frame_rate(16000).set_channels(1)
    samples = np.asarray(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
    result = model.generate(input=samples)
    if not result:
        return ""
    return result[0].get("text", "")
