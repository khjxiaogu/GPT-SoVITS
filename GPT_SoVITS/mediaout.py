# mediaout.py
"""语音生成结果的后处理工具。

主要职责：
1. 在代码里按中文标点（。？！……）切句，不再依赖 TTS 内部的切句逻辑；
2. 每句生成后立即切除首尾静音，再用 ASR 核对字数（标点不计）；
3. 字数差超过 20% 时按“最靠近句子中间的逗号”拆句重新生成；
4. 已不可再切、或字数差在 10%~20% 之间时重新生成一次，取字数差最小的版本；
5. 字数差在 2 个字以内视为无差异；
6. 把各段响度统一拉到平均响度（明显偏离的段直接剔除）。
"""

from dataclasses import dataclass
from pydub import AudioSegment
import numpy as np
import io
import math
import re
import unicodedata


@dataclass
class AudioPiece:
    """一段音频及其对应的原文，便于后续 ASR 校验。"""

    text: str
    segment: AudioSegment

# ---------------------------------------------------------------------------
# 文本切句
# ---------------------------------------------------------------------------

# 句末标点（省略号可能连续出现，如“……”，处理时会一并吞掉后续同类符号）
SENTENCE_END_CHARS = "。？！?!…"
# 用于二次拆句的逗号（全角、半角）
COMMA_CHARS = "，,"
# 仅由这些字符组成的片段视为空项
_PUNCT_CHARS = set(
    "。？！?!…，,、；;：:—~～.·\"'“”‘’（）()《》〈〉【】[]{} \t\r\n"
)

# 静音与 ASR 校验阈值
DEFAULT_SILENCE_DBFS = -45.0   # 低于该响度视为空白
DEFAULT_KEEP_SILENCE_MS = 50   # 切除首尾静音时保留的余量，避免切得太生硬
DEFAULT_MAX_SPLIT_DEPTH = 3    # 最多递归拆几层，避免无限拆分
DEFAULT_ASR_SPLIT_RATIO = 0.20  # 字数差超过 20% 时拆句
DEFAULT_ASR_RETRY_RATIO = 0.10  # 字数差超过 10% 时重新生成一次
DEFAULT_ASR_IGNORE_UNITS = 2.0  # 字数差在 2 个字以内视为无差异
DEFAULT_LOUDNESS_OUTLIER_DB = 12.0  # 低于响度中位数该值的段视为明显偏离，直接剔除


def _clean(text: str) -> str:
    return (text or "").strip()


def _meaningful_chars(text: str) -> str:
    """去掉标点与空白后的有效字符，用于判断片段是否为空。"""
    return "".join(ch for ch in text if ch not in _PUNCT_CHARS and not ch.isspace())


# 汉字 / 日文假名 / 韩文音节 / 西文单词（含数字）
_HAN_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_KANA_RE = re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff]")
_HANGUL_RE = re.compile(r"[\uac00-\ud7af\u1100-\u11ff]")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[.'’\-][A-Za-z0-9]+)*")


def count_text_units(text: str) -> float:
    """
    统计“字数”，用于比较输入文本与 ASR 识别结果。

    规则：
    - 英文/数字：一个单词算 1 个字；
    - 中文：一个汉字算 1 个字；
    - 日文：出现假名时按日文处理，假名和汉字合计每 2 个字符算 1 个字；
    - 韩文：一个音节算 1 个字。

    返回浮点数（日文可能出现 0.5）。
    """
    if not text:
        return 0.0

    text = unicodedata.normalize("NFKC", text)
    kana = len(_KANA_RE.findall(text))
    han = len(_HAN_RE.findall(text))
    hangul = len(_HANGUL_RE.findall(text))

    # 去掉汉/假名/韩文后再统计西文单词，避免重复计数
    rest = _HAN_RE.sub(" ", text)
    rest = _KANA_RE.sub(" ", rest)
    rest = _HANGUL_RE.sub(" ", rest)
    words = len(_WORD_RE.findall(rest))

    if kana > 0:
        # 日文：假名 + 汉字，两个字算一个字
        return (kana + han) / 2.0 + hangul + words
    return float(han + hangul + words)


def split_sentences(text: str) -> list:
    """
    按 。？！…… 等句末符号切句，保留标点，跳过空项。
    换行同样视为句子边界；只有标点或空白的片段会被丢弃。
    """
    if not text:
        return []

    sentences = []
    buf = []
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        if ch == "\r":
            i += 1
            continue

        if ch == "\n":
            sentence = _clean("".join(buf))
            if _meaningful_chars(sentence):
                sentences.append(sentence)
            buf = []
            i += 1
            continue

        buf.append(ch)
        if ch in SENTENCE_END_CHARS:
            # 连续标点（例如“？！”、“……”）归入同一句
            while i + 1 < n and text[i + 1] in SENTENCE_END_CHARS:
                i += 1
                buf.append(text[i])
            sentence = _clean("".join(buf))
            if _meaningful_chars(sentence):
                sentences.append(sentence)
            buf = []
        i += 1

    sentence = _clean("".join(buf))
    if _meaningful_chars(sentence):
        sentences.append(sentence)

    return sentences


def split_at_middle_comma(text: str, min_part_chars: int = 2):
    """
    在“最靠近句子中间”的逗号处把一句话拆成两句。
    前半句保留逗号（让 TTS 有自然的停顿），返回 (前一句, 后一句)；
    没有可用逗号时返回 None。
    """
    if not text:
        return None

    positions = [i for i, ch in enumerate(text) if ch in COMMA_CHARS]
    if not positions:
        return None

    center = (len(text) - 1) / 2.0
    for pos in sorted(positions, key=lambda p: abs(p - center)):
        left = _clean(text[: pos + 1])
        right = _clean(text[pos + 1:])
        if (
            len(_meaningful_chars(left)) >= min_part_chars
            and len(_meaningful_chars(right)) >= min_part_chars
        ):
            return left, right

    return None


# ---------------------------------------------------------------------------
# 音频空白检测
# ---------------------------------------------------------------------------


def _chunk_db(segment: AudioSegment, chunk_ms: float):
    """把音频按 chunk_ms 分块，返回每块的 dBFS、每块帧数与采样率。"""
    samples = np.asarray(segment.get_array_of_samples())
    if samples.size == 0:
        return np.empty(0, dtype=np.float64), 1, segment.frame_rate

    if segment.channels > 1:
        samples = samples.reshape(-1, segment.channels).mean(axis=1)

    max_amp = float(1 << (8 * segment.sample_width - 1))
    mono = samples.astype(np.float64) / max_amp

    chunk_frames = max(1, int(round(segment.frame_rate * chunk_ms / 1000.0)))
    n_chunks = mono.shape[0] // chunk_frames

    if n_chunks == 0:
        rms = math.sqrt(float(np.mean(mono ** 2)) + 1e-24)
        return np.array([20.0 * math.log10(max(rms, 1e-12))]), mono.shape[0], segment.frame_rate

    blocks = mono[: n_chunks * chunk_frames].reshape(n_chunks, chunk_frames)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1) + 1e-24)
    db = 20.0 * np.log10(np.maximum(rms, 1e-12))
    return db, chunk_frames, segment.frame_rate


def silence_stats(segment: AudioSegment, silence_dbfs: float = DEFAULT_SILENCE_DBFS,
                  chunk_ms: float = 20) -> dict:
    """
    统计音频中的空白（静音）情况。

    返回 dict：
      duration_ms     总时长
      silent_ms       空白总时长
      silent_ratio    空白占比（0~1）
      lead_silent_ms  开头连续空白时长
      trail_silent_ms 结尾连续空白时长
    """
    db, chunk_frames, frame_rate = _chunk_db(segment, chunk_ms)
    duration_ms = float(len(segment))

    if db.size == 0:
        return {
            "duration_ms": duration_ms,
            "silent_ms": 0.0,
            "silent_ratio": 0.0,
            "lead_silent_ms": 0.0,
            "trail_silent_ms": 0.0,
        }

    chunk_ms_actual = chunk_frames * 1000.0 / frame_rate
    silent = db < silence_dbfs
    silent_chunks = int(np.count_nonzero(silent))

    if silent_chunks == 0:
        lead_chunks = trail_chunks = 0
    elif silent_chunks == db.size:
        lead_chunks = trail_chunks = int(db.size)
    else:
        lead_chunks = int(np.argmax(~silent))
        trail_chunks = int(np.argmax(~silent[::-1]))

    return {
        "duration_ms": duration_ms,
        "silent_ms": silent_chunks * chunk_ms_actual,
        "silent_ratio": silent_chunks / float(db.size),
        "lead_silent_ms": lead_chunks * chunk_ms_actual,
        "trail_silent_ms": trail_chunks * chunk_ms_actual,
    }


def trim_silence(segment: AudioSegment, silence_dbfs: float = DEFAULT_SILENCE_DBFS,
                 chunk_ms: float = 10, keep_ms: float = DEFAULT_KEEP_SILENCE_MS) -> AudioSegment:
    """截掉音频首尾的空白，保留 keep_ms 的余量。"""
    if len(segment) == 0:
        return segment

    stats = silence_stats(segment, silence_dbfs=silence_dbfs, chunk_ms=chunk_ms)
    start = max(0, int(stats["lead_silent_ms"] - keep_ms))
    end = len(segment) - max(0, int(stats["trail_silent_ms"] - keep_ms))

    if end <= start:
        return AudioSegment.empty()
    return segment[start:end]


# ---------------------------------------------------------------------------
# 生成流程：切句 -> 生成 -> 切除首尾静音 -> ASR 核对 -> 拆句/重生成
# ---------------------------------------------------------------------------


def _synthesize_segment(text, synthesize_fn, silence_dbfs, keep_silence_ms):
    """合成一句，并立即切除首尾静音；生成失败返回 None。"""
    try:
        audio_tuple = synthesize_fn(text)
        segment = numpy_to_audiosegment(audio_tuple)
    except Exception as exc:
        print(f"[mediaout] 生成失败: {text!r} ({exc})")
        return None

    original_ms = len(segment)
    segment = trim_silence(segment, silence_dbfs=silence_dbfs, keep_ms=keep_silence_ms)
    if len(segment) != original_ms:
        print(f"[mediaout] 切除首尾静音: {original_ms}ms -> {len(segment)}ms")
    return segment


@dataclass
class _AsrCheck:
    transcript: str
    expected: float
    got: float
    diff: float
    ratio: float


def _check_text(text, segment, recognize_fn):
    """ASR 识别并按字数计算差距；没有 ASR 或识别失败时返回 None。"""
    if recognize_fn is None:
        return None

    expected = count_text_units(text)
    if len(segment) == 0:
        # 整段空白，直接当作完全不匹配，避免被“2 字以内不算差”放过
        print(f"[mediaout] ASR: 音频为空，跳过识别 | 原文={text!r} 字数={expected:g}")
        return _AsrCheck("", expected, 0.0, expected, 1.0)

    try:
        transcript = recognize_fn(segment) or ""
    except Exception as exc:
        print(f"[mediaout] ASR 识别失败，跳过字数校验: {text!r} ({exc})")
        return None

    got = count_text_units(transcript)
    diff = abs(got - expected)
    if expected <= 0 or diff <= DEFAULT_ASR_IGNORE_UNITS:
        ratio = 0.0
    else:
        ratio = diff / expected

    print(
        f"[mediaout] ASR: 原文={text!r} | 识别={transcript!r} | "
        f"字数={got:g}/{expected:g} | 差={diff:g}（{ratio:.0%}）"
    )

    return _AsrCheck(transcript, expected, got, diff, ratio)


def _retry_pick_best(text, segment, check, synthesize_fn, recognize_fn,
                     silence_dbfs, keep_silence_ms):
    """重新生成一次，在原版本与重生成版本中取字数差更小的一个。"""
    print(f"[mediaout] 字数差 {check.diff:g}（{check.ratio:.0%}），重新生成一次: {text!r}")
    retry_segment = _synthesize_segment(text, synthesize_fn, silence_dbfs, keep_silence_ms)

    best_segment, best_diff = segment, check.diff
    if retry_segment is not None and len(retry_segment) > 0:
        retry_check = _check_text(text, retry_segment, recognize_fn)
        if retry_check is not None:
            print(f"[mediaout] 重新生成后字数差 {retry_check.diff:g}（{retry_check.ratio:.0%}）")
            if retry_check.diff < best_diff:
                best_segment, best_diff = retry_segment, retry_check.diff
                print(f"[mediaout] 采用重新生成的版本: {text!r}")

    if len(best_segment) == 0:
        print(f"[mediaout] 两次生成都没有有效音频，剔除: {text!r}")
        return []
    return [AudioPiece(text=text, segment=best_segment)]


def _generate_one(text, synthesize_fn, recognize_fn, depth, max_split_depth,
                  min_part_chars, silence_dbfs, keep_silence_ms):
    text = _clean(text)
    if not _meaningful_chars(text):
        return []

    segment = _synthesize_segment(text, synthesize_fn, silence_dbfs, keep_silence_ms)
    if segment is None:
        return []

    check = _check_text(text, segment, recognize_fn)
    if check is None:
        # 没有 ASR 可用时，只保留切除静音后的音频
        return [AudioPiece(text=text, segment=segment)] if len(segment) > 0 else []

    if check.ratio <= DEFAULT_ASR_RETRY_RATIO:
        return [AudioPiece(text=text, segment=segment)] if len(segment) > 0 else []

    # 字数差超过 20%：优先按最靠近中间的逗号拆成两句重新生成
    if check.ratio > DEFAULT_ASR_SPLIT_RATIO and depth < max_split_depth:
        parts = split_at_middle_comma(text, min_part_chars=min_part_chars)
        if parts:
            print(
                f"[mediaout] ASR 字数差 {check.diff:g}（{check.ratio:.0%}）超过 "
                f"{DEFAULT_ASR_SPLIT_RATIO:.0%}，拆句重试: {parts[0]!r} | {parts[1]!r}"
            )
            left = _generate_one(
                parts[0], synthesize_fn, recognize_fn, depth + 1, max_split_depth,
                min_part_chars, silence_dbfs, keep_silence_ms,
            )
            right = _generate_one(
                parts[1], synthesize_fn, recognize_fn, depth + 1, max_split_depth,
                min_part_chars, silence_dbfs, keep_silence_ms,
            )
            if left and right:
                return left + right
            print("[mediaout] 拆句后生成不完整，退回整句重新生成")

    # 已不可再切，或字数差在 10%~20% 之间：重新生成一次，取字数差最小的版本
    return _retry_pick_best(
        text, segment, check, synthesize_fn, recognize_fn, silence_dbfs, keep_silence_ms
    )


def generate_audio_segments(
    text: str,
    synthesize_fn,
    recognize_fn=None,
    *,
    max_split_depth: int = DEFAULT_MAX_SPLIT_DEPTH,
    min_part_chars: int = 2,
    silence_dbfs: float = DEFAULT_SILENCE_DBFS,
    keep_silence_ms: float = DEFAULT_KEEP_SILENCE_MS,
) -> list:
    """
    按标点切句后逐句生成：每句生成完立即切除首尾静音并跑 ASR 核对字数，
    字数差超过 20% 时按中间逗号拆句；不可再切或字数差在 10%~20% 时
    重新生成一次并取字数差最小的版本。

    :param text:          原始文本
    :param synthesize_fn: 合成回调，签名 synthesize_fn(text) -> (sample_rate, np.ndarray)
    :param recognize_fn:  ASR 回调，签名 recognize_fn(segment) -> str；为 None 时跳过校验
    :return: AudioPiece 列表（含原文与音频），按句子顺序排列
    """
    pieces = []
    for sentence in split_sentences(text):
        pieces.extend(
            _generate_one(
                sentence, synthesize_fn, recognize_fn, 0, max_split_depth,
                min_part_chars, silence_dbfs, keep_silence_ms,
            )
        )
    return pieces


# ---------------------------------------------------------------------------
# 响度统一：把所有片段拉到平均响度
# ---------------------------------------------------------------------------


def _segment_dbfs(segment):
    """整段音频的 RMS 响度（dBFS），静音/空音频返回 None。"""
    if len(segment) == 0:
        return None
    try:
        db = float(segment.dBFS)
    except Exception:
        return None
    if not math.isfinite(db):
        return None
    return db


def normalize_loudness(pieces, target_dbfs=None, outlier_db: float = DEFAULT_LOUDNESS_OUTLIER_DB):
    """
    计算所有片段的平均响度，再用增益把每段拉到该响度（接近即可）。
    明显偏离的段（近乎静音等）直接剔除：既不参与平均响度，也不做增益。

    :param pieces:      AudioPiece 列表，会被原地修改并返回
    :param target_dbfs: 指定目标响度；为 None 时取剔除离群值后的平均
    :param outlier_db:  低于中位数该值（dB）的片段视为明显偏离
    """
    levels = []
    for piece in pieces:
        db = _segment_dbfs(piece.segment)
        if db is not None:
            levels.append(db)

    if not levels:
        return pieces

    if target_dbfs is None:
        median = float(np.median(levels))
        kept = [db for db in levels if db >= median - outlier_db]
        if not kept:
            kept = levels
        target = float(np.mean(kept))
        excluded = [db for db in levels if db < median - outlier_db]
        if excluded:
            print(
                f"[mediaout] 剔除 {len(excluded)} 段明显偏离的音频"
                f"（低于中位数 {outlier_db:.0f}dB），不参与响度统一"
            )
    else:
        target = float(target_dbfs)
        excluded = []

    print(f"[mediaout] 平均响度 {target:.1f} dBFS，统一 {len(levels) - len(excluded)} 段音频的响度")

    for piece in pieces:
        db = _segment_dbfs(piece.segment)
        if db is None:
            continue
        if target_dbfs is None and db < median - outlier_db:
            continue  # 近乎静音等明显偏离的段直接跳过
        gain = target - db
        if abs(gain) >= 0.05:
            piece.segment = piece.segment.apply_gain(gain)

    return pieces


# ---------------------------------------------------------------------------
# 基础音频工具
# ---------------------------------------------------------------------------


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
