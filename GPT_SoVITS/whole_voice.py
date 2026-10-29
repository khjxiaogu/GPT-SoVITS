# whole_voice.py
"""WebUI“整段语音”页签。

功能：
1. 按 localemp 的请求格式（chara + content[{emotion,text}]）手动录入分段，
   一键生成整段语音；
2. 用大语言模型（OpenAI 兼容 /chat/completions）根据剧情文本自动生成分段 JSON，
   支持流式返回与“思考”参数，所有 API 参数均可在页面上配置；
3. 生成流程复用 mediaout：逐句切分、切静音、ASR 字数校验、响度统一。
"""

import atexit
import json
import math
import os
import re
import shutil
import tempfile
import time
import traceback
import zipfile

import gradio as gr
import requests

import mediaout
import llm_config
from asr_utils import get_asr_model, recognize_audio

# 全部使用项目内相对目录（以项目根目录为工作目录），方便整体迁移
KH_MODELS_DIR = "kh_models"
OUTPUT_DIR = os.path.join("output", "whole_voice")

EMOTIONS = ["开心", "吃惊", "恐惧", "难过", "生气", "中立", "厌恶"]
REF_EXTS = (".mp3", ".wav", ".ogg", ".flac", ".m4a")
DEFAULT_PROMPT_PATH = os.path.join("prompts", "voice-tags.txt")
DEFAULT_THINKING_JSON = '{"enable_thinking": true}'

# 常用接口参数模板（API Key 不预置，需自己填写）
API_TEMPLATES = {
    "自定义": None,
    "MiMo（小米）": {
        "base_url": "https://api.xiaomimimo.com/v1",
        "model": "mimo-v2.6-flash",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 4096,
        "token_field": "max_completion_tokens",
        "thinking": True,
        "thinking_json": '{"thinking": {"type": "enabled"}}',
        "extra_json": "{}",
    },
    "火山方舟（豆包）": {
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
        "model": "doubao-seed-1-6-lite",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4096,
        "token_field": "max_tokens",
        "thinking": True,
        "thinking_json": '{"thinking": {"type": "enabled"}}',
        "extra_json": "{}",
    },
    "DeepSeek": {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-flash",
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": 4096,
        "token_field": "max_tokens",
        "thinking": True,
        "thinking_json": '{"thinking": {"type": "enabled"}}',
        "extra_json": "{}",
    },
}
TOKEN_FIELDS = ["max_tokens", "max_completion_tokens", "不发送"]


# ---------------------------------------------------------------------------
# 角色（目录或 zip）/ 参考音频
# ---------------------------------------------------------------------------

ZIP_PREFIX = "zip::"
_ZIP_CACHE = {}
_EXTRACTED_DIRS = []


def _cleanup_extracted():
    for folder in _EXTRACTED_DIRS:
        shutil.rmtree(folder, ignore_errors=True)


atexit.register(_cleanup_extracted)


def _read_meta_file(folder):
    path = os.path.join(folder, "meta.json")
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8-sig") as file:
            meta = json.load(file)
        return meta if isinstance(meta, dict) else None
    except Exception:
        traceback.print_exc()
        return None


def _zip_member_name(info):
    """修正 Windows 下非 UTF-8 标记 zip 的中文文件名。"""
    name = info.filename
    if not (info.flag_bits & 0x800):
        try:
            name = name.encode("cp437").decode("gbk")
        except Exception:
            pass
    return name.replace("\\", "/")


def _zip_meta_members(zip_file):
    members = []
    for info in zip_file.infolist():
        name = _zip_member_name(info).strip("/")
        if name == "meta.json":
            members.append((0, name, info))
        elif name.endswith("/meta.json") and name.count("/") == 1:
            members.append((1, name, info))
    members.sort(key=lambda item: (item[0], item[1]))
    return members


def _read_meta_from_zip(zip_path):
    try:
        with zipfile.ZipFile(zip_path) as zip_file:
            members = _zip_meta_members(zip_file)
            if not members:
                return None
            meta = json.loads(zip_file.read(members[0][2]).decode("utf-8-sig"))
            return meta if isinstance(meta, dict) else None
    except Exception:
        traceback.print_exc()
        return None


def _extract_zip(zip_path):
    """把角色 zip 解压到临时目录，同一个 zip 只解压一次。"""
    stat = os.stat(zip_path)
    cache_key = (os.path.abspath(zip_path), stat.st_mtime_ns, stat.st_size)
    cached = _ZIP_CACHE.get(cache_key)
    if cached and os.path.isdir(cached):
        return cached

    target = tempfile.mkdtemp(prefix="gptsovits_chara_")
    _EXTRACTED_DIRS.append(target)
    with zipfile.ZipFile(zip_path) as zip_file:
        for info in zip_file.infolist():
            if info.is_dir():
                continue
            name = _zip_member_name(info)
            norm = os.path.normpath(name)
            if os.path.isabs(norm) or norm.startswith("..") or re.match(r"^[A-Za-z]:", norm):
                print(f"[whole_voice] 跳过压缩包内的不安全路径: {name}")
                continue
            dest = os.path.join(target, norm)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with zip_file.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)

    _ZIP_CACHE[cache_key] = target
    return target


def _find_character_root(root):
    """解压后找到含 meta.json 的目录（压缩包根目录或唯一子目录）。"""
    if os.path.isfile(os.path.join(root, "meta.json")):
        return root
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        if os.path.isdir(folder) and os.path.isfile(os.path.join(folder, "meta.json")):
            return folder
    return None


def _display_name(key, meta, fallback):
    name = str(meta.get("name") or "").strip()
    return name or fallback


def list_character_entries():
    """
    遍历 kh_models：
    - 子目录必须含 meta.json 才算角色；
    - *.zip 压缩包需内含 meta.json；
    - 显示名优先用 meta.json 的 name 字段。
    """
    entries = []
    if not os.path.isdir(KH_MODELS_DIR):
        return entries

    for name in sorted(os.listdir(KH_MODELS_DIR)):
        path = os.path.join(KH_MODELS_DIR, name)
        if os.path.isdir(path):
            meta = _read_meta_file(path)
            if meta is None:
                continue
            entries.append({
                "key": name,
                "label": _display_name(name, meta, name),
                "source": name,
                "is_zip": False,
            })
        elif name.lower().endswith(".zip") and os.path.isfile(path):
            meta = _read_meta_from_zip(path)
            if meta is None:
                print(f"[whole_voice] 跳过没有 meta.json 的压缩包: {name}")
                continue
            stem = os.path.splitext(name)[0]
            entries.append({
                "key": ZIP_PREFIX + name,
                "label": _display_name(name, meta, stem),
                "source": name,
                "is_zip": True,
            })

    # 显示名重复时补上来源，避免下拉里分不清
    counts = {}
    for entry in entries:
        counts[entry["label"]] = counts.get(entry["label"], 0) + 1
    for entry in entries:
        if counts[entry["label"]] > 1:
            entry["label"] = f"{entry['label']}（{entry['source']}）"

    return entries


def list_characters():
    """角色键列表（兼容旧调用）。"""
    return [entry["key"] for entry in list_character_entries()]


def resolve_character(key):
    """把下拉值解析成 (角色目录, meta)；zip 会先解压到临时目录。"""
    if not key:
        raise ValueError("请先选择角色。")

    if key.startswith(ZIP_PREFIX):
        zip_name = key[len(ZIP_PREFIX):]
        zip_path = os.path.join(KH_MODELS_DIR, zip_name)
        if not os.path.isfile(zip_path):
            raise ValueError(f"角色压缩包不存在：{zip_path}")
        folder = _find_character_root(_extract_zip(zip_path))
        if folder is None:
            raise ValueError(f"压缩包里没有找到 meta.json：{zip_name}")
    else:
        folder = os.path.join(KH_MODELS_DIR, key)
        if not os.path.isdir(folder):
            raise ValueError(f"角色目录不存在：{folder}")

    meta = _read_meta_file(folder)
    if meta is None:
        raise ValueError(f"角色目录缺少 meta.json：{folder}")
    return folder, meta


def find_reference(folder, emotion):
    """按 【emotion】.mp3 / .wav … 在角色目录里查找参考音频。"""
    for ext in REF_EXTS:
        path = os.path.join(folder, f"【{emotion}】{ext}")
        if os.path.isfile(path):
            return path
    return None


def load_persona(chara):
    """读取角色目录下的 charaset.txt；没有则返回空字符串。"""
    if not chara:
        return ""
    try:
        folder, _ = resolve_character(chara)
    except Exception:
        return ""
    path = os.path.join(folder, "charaset.txt")
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8-sig") as file:
            return file.read().strip()
    except Exception:
        traceback.print_exc()
        return ""


def build_user_content(persona, story):
    """按约定拼接【角色人设】+ 人设内容 +【待处理语音文本】+ 文本。"""
    persona = (persona or "").strip()
    story = (story or "").strip()
    persona_block = "【角色人设】\n" + persona if persona else "【角色人设】"
    return persona_block + "\n\n【待处理语音文本】\n" + story


# ---------------------------------------------------------------------------
# 分段数据校整
# ---------------------------------------------------------------------------


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return str(value).strip()


def normalize_segments(raw):
    """
    把表格行 / 字典列表 / {"data": [...]} 统一成 [{"emotion":..., "text":...}]。
    空文本行直接跳过，非法情感标签抛 ValueError。
    """
    if raw is None:
        return []

    if isinstance(raw, dict):
        for key in ("data", "segments", "content", "result", "items"):
            value = raw.get(key)
            if isinstance(value, list):
                raw = value
                break
        else:
            raw = [raw] if ("emotion" in raw and "text" in raw) else []

    if not isinstance(raw, list):
        raise ValueError("分段数据不是数组")

    segments = []
    for index, item in enumerate(raw, 1):
        if isinstance(item, (list, tuple)):
            if len(item) < 2:
                continue
            emotion, text = item[0], item[1]
        elif isinstance(item, dict):
            emotion, text = item.get("emotion"), item.get("text")
        else:
            raise ValueError(f"第 {index} 段格式不对，应为 emotion/text")

        emotion = _cell(emotion)
        text = _cell(text)
        if not text:
            continue
        if emotion not in EMOTIONS:
            raise ValueError(
                f"第 {index} 段的情感标签“{emotion}”无效，只能使用：{'、'.join(EMOTIONS)}"
            )
        segments.append({"emotion": emotion, "text": text})

    return segments


def segments_to_rows(segments):
    return [[item["emotion"], item["text"]] for item in segments]


def segments_to_json(segments):
    return json.dumps({"data": segments}, ensure_ascii=False, indent=2)


def table_to_json(rows):
    try:
        segments = normalize_segments(rows)
    except ValueError as exc:
        return gr.update(), f"❌ {exc}"
    return segments_to_json(segments), f"✅ 已导出 {len(segments)} 段"


def json_to_table(raw):
    try:
        segments = normalize_segments(_loads_json_loose(raw))
    except Exception as exc:
        return gr.update(), f"❌ JSON 解析失败：{exc}"
    if not segments:
        return [["中立", ""]], "⚠️ JSON 中没有有效分段"
    return segments_to_rows(segments), f"✅ 已导入 {len(segments)} 段"


# ---------------------------------------------------------------------------
# 大语言模型：OpenAI 兼容 /chat/completions（不依赖 openai 包）
# ---------------------------------------------------------------------------


def _loads_json_loose(text):
    """从模型返回里提取 JSON，兼容 ```json 代码块和前后说明文字。"""
    if not text or not text.strip():
        raise ValueError("返回内容为空")
    cleaned = text.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S | re.I)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    starts = [pos for pos in (cleaned.find("["), cleaned.find("{")) if pos >= 0]
    if not starts:
        raise ValueError("返回内容中没有找到 JSON")
    start = min(starts)
    end = max(cleaned.rfind("]"), cleaned.rfind("}"))
    if end <= start:
        raise ValueError("返回内容中的 JSON 不完整")
    return json.loads(cleaned[start:end + 1])


def _content_text(content):
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(part.get("text") or part.get("content") or "")
        return "".join(parts)
    return str(content)


def _extract_message_text(data):
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"API 返回中没有 choices：{json.dumps(data, ensure_ascii=False)[:500]}")
    message = choices[0].get("message") or {}
    text = _content_text(message.get("content"))
    if not text:
        raise RuntimeError("API 返回的 message.content 为空")
    return text


def _read_sse(response):
    pieces = []
    for raw_line in response.iter_lines(decode_unicode=False):
        if not raw_line:
            continue
        # 统一按 UTF-8 解码，避免服务端 Content-Type 未声明 charset 时按 latin-1 处理
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        if not payload:
            continue
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if "error" in obj:
            raise RuntimeError(f"API 流式返回错误：{obj['error']}")
        choices = obj.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or choice.get("message") or {}
        text = _content_text(delta.get("content"))
        if text:
            pieces.append(text)

    result = "".join(pieces)
    if not result:
        raise RuntimeError("流式响应中没有解析到内容")
    return result


def _parse_json_object(raw, label):
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}不是合法 JSON：{exc}")
    if not isinstance(obj, dict):
        raise ValueError(f"{label}必须是 JSON 对象")
    return obj


def run_completion(
    base_url,
    api_key,
    model_name,
    temperature,
    top_p,
    max_tokens,
    token_field,
    presence_penalty,
    frequency_penalty,
    seed,
    stop,
    timeout,
    stream,
    thinking,
    thinking_json,
    extra_json,
    system_prompt,
    persona,
    story,
    system_as_user,
):
    """调用 OpenAI 兼容的 /chat/completions，返回模型文本。"""
    base = (base_url or "").strip()
    model = (model_name or "").strip()
    prompt = (system_prompt or "").strip()
    story = (story or "").strip()

    if not base:
        raise ValueError("请填写 API Base URL")
    if not model:
        raise ValueError("请填写模型名称")
    if not prompt:
        raise ValueError("系统提示词为空")
    if not story:
        raise ValueError("剧情文本为空")

    user_content = build_user_content(persona, story)
    if system_as_user:
        messages = [{"role": "user", "content": prompt + "\n\n" + user_content}]
    else:
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ]

    body = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stream": bool(stream),
        "presence_penalty": float(presence_penalty),
        "frequency_penalty": float(frequency_penalty),
    }
    if max_tokens and int(max_tokens) > 0 and token_field != "不发送":
        body[token_field or "max_tokens"] = int(max_tokens)
    if int(seed) >= 0:
        body["seed"] = int(seed)

    stop_list = [item.strip() for item in re.split(r"[|,，]", stop or "") if item.strip()]
    if stop_list:
        body["stop"] = stop_list

    if thinking:
        body.update(_parse_json_object(thinking_json, "思考参数"))
    body.update(_parse_json_object(extra_json, "额外请求参数"))

    headers = {"Content-Type": "application/json"}
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key.strip()}"

    url = base.rstrip("/")
    if not re.search(r"/chat/completions$", url):
        url += "/chat/completions"

    response = requests.post(
        url,
        headers=headers,
        json=body,
        timeout=float(timeout),
        stream=bool(stream),
    )
    if response.status_code >= 400:
        raise RuntimeError(f"API 返回 {response.status_code}：{response.text[:800]}")

    if stream:
        return _read_sse(response)
    return _extract_message_text(response.json())


def generate_segments_from_llm(**kwargs):
    raw = run_completion(**kwargs)
    try:
        payload = _loads_json_loose(raw)
        segments = normalize_segments(payload)
    except Exception as exc:
        raise ValueError(f"JSON 解析失败：{exc}\n\n模型原始返回：\n{raw[:1500]}")
    if not segments:
        raise ValueError("模型没有返回任何有效分段")
    return segments


# ---------------------------------------------------------------------------
# 整段语音合成
# ---------------------------------------------------------------------------


def _make_synthesizer(tts_fn, ref_path, text_language, top_k, top_p, temperature, speed):
    def synthesize(text):
        generator = tts_fn(
            ref_wav_path=ref_path,
            prompt_text=None,
            prompt_language="中文",
            text=text,
            text_language=text_language,
            how_to_cut="不切",
            top_k=int(top_k),
            top_p=float(top_p),
            temperature=float(temperature),
            speed=float(speed),
        )
        return next(generator)

    return synthesize


# ---------------------------------------------------------------------------
# 页签 UI
# ---------------------------------------------------------------------------


def synthesize_whole(
    chara_name,
    segments_data,
    *,
    text_language,
    top_k,
    top_p,
    temperature,
    speed,
    sentence_gap_ms,
    segment_gap_ms,
    use_asr,
    tts_fn,
    change_sovits_fn,
    change_gpt_fn,
    progress_cb=None,
):
    """生成整段语音，返回 (mp3 路径, 状态文本)；参数问题抛 ValueError。"""
    if not chara_name:
        raise ValueError("请先选择角色。")
    if not segments_data:
        raise ValueError("没有有效分段：请填写 emotion 与 text。")

    folder, meta = resolve_character(chara_name)
    if not meta.get("sovits") or not meta.get("gpt"):
        raise ValueError("meta.json 缺少 sovits 或 gpt 配置，必须提供才能加载角色模型")

    # meta.json 里的模型配置必须应用
    list(change_sovits_fn(meta["sovits"], "中文", "中文"))
    change_gpt_fn(meta["gpt"])

    notes = []
    recognize_fn = None
    if use_asr:
        if get_asr_model() is not None:
            recognize_fn = recognize_audio
        else:
            notes.append("ASR 模型不可用，已跳过字数校验")

    generated = []
    total = len(segments_data)
    for index, segment in enumerate(segments_data):
        emotion, text = segment["emotion"], segment["text"]
        reference = find_reference(folder, emotion)
        if reference is None:
            raise ValueError(
                f"角色 {chara_name} 缺少情感参考音频：【{emotion}】.(mp3/wav/ogg)"
            )

        if progress_cb:
            progress_cb(index / total, f"生成 {index + 1}/{total}：{emotion}")
        synthesize = _make_synthesizer(
            tts_fn, reference, text_language, top_k, top_p, temperature, speed
        )
        try:
            pieces = mediaout.generate_audio_segments(text, synthesize, recognize_fn)
        except Exception as exc:
            raise RuntimeError(f"第 {index + 1} 段生成失败：{exc}") from exc
        if not pieces:
            notes.append(f"第 {index + 1} 段没有生成出有效音频")
            continue
        generated.append(pieces)

    if not generated:
        raise RuntimeError("所有分段都没有生成出音频。")

    all_pieces = [piece for pieces in generated for piece in pieces]
    mediaout.normalize_loudness(all_pieces)

    audio_segments = [
        mediaout.concat_with_silence(
            [piece.segment for piece in pieces], silence_ms=int(sentence_gap_ms)
        )
        for pieces in generated
    ]
    combined = mediaout.concat_with_silence(audio_segments, silence_ms=int(segment_gap_ms))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    display = _display_name(chara_name, meta, chara_name.split(ZIP_PREFIX)[-1])
    display = re.sub(r'[\\/:*?"<>|]+', "_", display) or "voice"
    filename = f"{display}_{time.strftime('%Y%m%d_%H%M%S')}.mp3"
    output_path = os.path.join(OUTPUT_DIR, filename)
    combined.export(output_path, format="mp3")

    status = (
        f"✅ 完成：{len(generated)} 段 / {len(all_pieces)} 句 / "
        f"{len(combined) / 1000:.1f} 秒\n\n输出文件：`{output_path}`"
    )
    if notes:
        status += "\n\n" + "\n".join(f"- {note}" for note in notes)
    return output_path, status


def build_ui(*, get_tts_wav_fn, change_sovits_fn, change_gpt_fn,
             language_choices, default_language="中文"):
    entries = list_character_entries()
    choices = [(entry["label"], entry["key"]) for entry in entries]
    default_char = choices[0][1] if choices else None
    default_prompt = ""
    if os.path.isfile(DEFAULT_PROMPT_PATH):
        try:
            with open(DEFAULT_PROMPT_PATH, "r", encoding="utf-8") as file:
                default_prompt = file.read()
        except Exception:
            traceback.print_exc()

    saved_api = llm_config.load_config()
    api_key_state = {"value": saved_api.get("api_key", "")}

    def _saved(key, default):
        value = saved_api.get(key)
        return default if value is None else value

    with gr.Row():
        # ----------------------------- 左侧：手动分段 -----------------------------
        with gr.Column(scale=11):
            gr.Markdown("### 手动分段（对应 localemp 的 chara + content）")

            with gr.Row():
                chara = gr.Dropdown(
                    label="角色",
                    choices=choices,
                    value=default_char,
                    allow_custom_value=False,
                    scale=7,
                )
                refresh_chara = gr.Button("刷新角色", scale=1)

            gr.Markdown("情感标签只能使用：**" + "、".join(EMOTIONS) + "**")

            segments = gr.Dataframe(
                headers=["emotion", "text"],
                datatype=["str", "str"],
                type="array",
                value=[["中立", ""]],
                row_count=(1, "dynamic"),
                col_count=(2, "fixed"),
                interactive=True,
                wrap=True,
                label="分段表（每行一段）",
            )

            with gr.Row():
                add_row = gr.Button("＋ 添加一行")
                del_row = gr.Button("－ 删除末行")
                export_json = gr.Button("表格 → JSON")

            with gr.Accordion("分段 JSON（导入 / 导出）", open=False):
                segments_json = gr.Textbox(
                    label="分段 JSON（可粘贴后点击“JSON → 表格”）",
                    value="[]",
                    lines=12,
                    max_lines=24,
                )
                import_json = gr.Button("JSON → 表格")

            with gr.Accordion("合成参数", open=True):
                with gr.Row():
                    text_language = gr.Dropdown(
                        label="语种",
                        choices=list(language_choices),
                        value=default_language,
                        scale=2,
                    )
                    speed = gr.Slider(0.6, 1.65, value=1.0, step=0.05, label="语速")
                with gr.Row():
                    top_k = gr.Slider(1, 100, value=30, step=1, label="top_k")
                    top_p = gr.Slider(0, 1, value=0.6, step=0.05, label="top_p")
                    temperature = gr.Slider(0, 1, value=0.6, step=0.05, label="temperature")
                with gr.Row():
                    sentence_gap = gr.Slider(0, 1500, value=300, step=50, label="句间静音(ms)")
                    segment_gap = gr.Slider(0, 3000, value=500, step=50, label="段间静音(ms)")
                use_asr = gr.Checkbox(label="启用 ASR 字数校验（较慢）", value=True)

            generate_button = gr.Button("生成整段语音", variant="primary", size="lg")
            audio_output = gr.Audio(label="整段语音", type="filepath", interactive=False)
            generate_status = gr.Markdown()

        # ----------------------------- 右侧：LLM 自动分段 -----------------------------
        with gr.Column(scale=9):
            gr.Markdown("### 大语言模型自动分段")

            with gr.Accordion("角色人设", open=False):
                with gr.Row():
                    persona = gr.Textbox(
                        label="内容",
                        value=load_persona(default_char),
                        lines=5,
                        max_lines=12,
                        scale=5,
                    )
                    persona_load_button = gr.Button("载入角色人设", scale=1)

            story = gr.Textbox(
                label="剧情文本（需包含“待处理角色话语”标记）",
                lines=10,
                max_lines=20,
                placeholder="把包含角色台词的剧情粘贴到这里……",
            )

            with gr.Accordion("系统提示词", open=False):
                system_prompt = gr.Textbox(
                    label="内容",
                    value=default_prompt,
                    lines=8,
                    max_lines=18,
                )
                with gr.Row():
                    prompt_path = gr.Textbox(
                        label="提示词文件路径",
                        value=DEFAULT_PROMPT_PATH,
                        scale=5,
                    )
                    load_prompt_button = gr.Button("加载提示词", scale=1)

            with gr.Accordion("API 参数", open=False):
                with gr.Row():
                    template = gr.Dropdown(
                        label="参数模板",
                        choices=list(API_TEMPLATES.keys()),
                        value="自定义",
                        scale=2,
                    )
                    gr.Markdown(
                        "选择模板会自动填充 Base URL、模型和思考参数；"
                        "火山方舟的模型名也可改成自己的推理接入点 ID（ep-...）。"
                    )
                with gr.Row():
                    base_url = gr.Textbox(
                        label="Base URL",
                        placeholder="https://host/v1",
                        value=_saved("base_url", ""),
                        scale=3,
                    )
                    api_key = gr.Textbox(
                        label="API Key",
                        type="password",
                        value="",
                        placeholder=(
                            "已保存（留空自动使用）"
                            if api_key_state["value"]
                            else "未保存"
                        ),
                        scale=2,
                    )
                gr.Markdown(
                    "已保存的 API Key 不会回显到页面：留空即自动使用；"
                    "输入新的 Key 会覆盖保存的 Key。"
                )
                with gr.Row():
                    model_name = gr.Textbox(
                        label="模型名称",
                        placeholder="model-name",
                        value=_saved("model_name", ""),
                        scale=3,
                    )
                    saved_token_field = _saved("token_field", "max_tokens")
                    token_field = gr.Dropdown(
                        label="token 上限字段",
                        choices=TOKEN_FIELDS,
                        value=saved_token_field if saved_token_field in TOKEN_FIELDS else "max_tokens",
                        scale=2,
                    )
                    timeout = gr.Number(
                        label="超时(秒)", value=_saved("timeout", 180), precision=0, scale=1
                    )
                with gr.Row():
                    llm_temperature = gr.Slider(
                        0, 2, value=_saved("temperature", 0.7), step=0.05, label="temperature"
                    )
                    llm_top_p = gr.Slider(
                        0, 1, value=_saved("top_p", 0.9), step=0.05, label="top_p"
                    )
                    max_tokens = gr.Number(
                        label="token 上限", value=_saved("max_tokens", 2048), precision=0
                    )
                with gr.Row():
                    presence_penalty = gr.Slider(
                        -2, 2, value=_saved("presence_penalty", 0), step=0.1,
                        label="presence_penalty",
                    )
                    frequency_penalty = gr.Slider(
                        -2, 2, value=_saved("frequency_penalty", 0), step=0.1,
                        label="frequency_penalty",
                    )
                with gr.Row():
                    seed = gr.Number(
                        label="seed（-1 表示不传）", value=_saved("seed", -1), precision=0
                    )
                    stop = gr.Textbox(label="stop（用 | 分隔）", value=_saved("stop", ""))
                with gr.Row():
                    stream = gr.Checkbox(label="流式返回", value=_saved("stream", True))
                    thinking = gr.Checkbox(label="开启思考", value=_saved("thinking", True))
                    system_as_user = gr.Checkbox(
                        label="提示词作为 user 消息", value=_saved("system_as_user", False)
                    )
                thinking_json = gr.Textbox(
                    label="思考参数（合并进请求体）",
                    value=_saved("thinking_json", DEFAULT_THINKING_JSON),
                    lines=2,
                )
                extra_json = gr.Textbox(
                    label="额外请求参数 JSON（会覆盖同名参数）",
                    value=_saved("extra_json", "{}"),
                    lines=3,
                )
                with gr.Row():
                    remember_key = gr.Checkbox(
                        label="记住 API Key（Windows 加密保存，仅当前用户可解密）",
                        value=_saved("save_api_key", True),
                        scale=3,
                    )
                    save_config_button = gr.Button("保存 API 配置", scale=1)
                    clear_config_button = gr.Button("删除已保存配置", scale=1)

                gr.Markdown(
                    "配置保存到项目内 `configs/llm_config.json`："
                    "除 API Key 外为明文，API Key 用 Windows DPAPI 加密。"
                )

            llm_button = gr.Button("生成分段 JSON", variant="primary", size="lg")
            llm_status = gr.Markdown()

    with gr.Accordion("模型文件夹制作说明", open=False):
        gr.Markdown(
            "在 `kh_models/` 下为每个角色建一个文件夹，或打包成 `角色.zip`。\n\n"
            "```\n"
            "角色名/\n"
            "├── meta.json          # 必填\n"
            "├── 【开心】.mp3        # 7 种情感各一份参考音频\n"
            "├── 【吃惊】.mp3\n"
            "├── 【恐惧】.mp3\n"
            "├── 【难过】.mp3\n"
            "├── 【生气】.mp3\n"
            "├── 【中立】.mp3\n"
            "├── 【厌恶】.mp3\n"
            "└── charaset.txt       # 可选，角色人设\n"
            "```\n\n"
            "**meta.json 示例**\n"
            "```json\n"
            "{\n"
            '  "name": "角色显示名",\n'
            '  "sovits": "SoVITS_weights_v2ProPlus/xxx.pth",\n'
            '  "gpt": "GPT_weights_v2ProPlus/xxx.ckpt"\n'
            "}\n"
            "```\n\n"
            "- `name`：角色列表里的显示名；不填则用文件夹名或压缩包名。\n"
            "- `sovits` / `gpt`：模型权重路径（相对项目根目录），生成时自动加载。\n"
            "- 参考音频建议 3~10 秒，也支持 `.wav` / `.ogg` / `.flac` / `.m4a`。\n"
            "- `charaset.txt`：可选，UTF-8 文本，会自动读入“角色人设”。\n"
            "- zip 结构相同，`meta.json` 在压缩包根目录或唯一一层子目录中均可。\n"
            "- 没有 `meta.json` 的目录或压缩包不会出现在角色列表。\n"
        )

    # ----------------------------- 事件绑定 -----------------------------
    def _add_row(rows):
        rows = [list(row) for row in (rows or [])]
        rows.append(["中立", ""])
        return rows

    def _del_row(rows):
        rows = [list(row) for row in (rows or [])]
        return rows[:-1] if rows else [["中立", ""]]

    def _generate(chara_name, rows, language, k, p, temp, spd, sent_gap, seg_gap,
                  asr_enabled, progress=gr.Progress()):
        try:
            segments_data = normalize_segments(rows)
            return synthesize_whole(
                chara_name,
                segments_data,
                text_language=language,
                top_k=k,
                top_p=p,
                temperature=temp,
                speed=spd,
                sentence_gap_ms=sent_gap,
                segment_gap_ms=seg_gap,
                use_asr=asr_enabled,
                tts_fn=get_tts_wav_fn,
                change_sovits_fn=change_sovits_fn,
                change_gpt_fn=change_gpt_fn,
                progress_cb=lambda frac, desc: progress(frac, desc=desc),
            )
        except Exception as exc:
            traceback.print_exc()
            return None, f"❌ {exc}"

    def _load_prompt(path):
        try:
            with open(path, "r", encoding="utf-8") as file:
                content = file.read()
            return content, f"✅ 已加载提示词：{path}"
        except Exception as exc:
            return gr.update(), f"❌ 加载提示词失败：{exc}"

    def _api_payload(base, key, model, temp, p, tokens, token_field_name,
                     presence, frequency, sd, stp, tmo, is_stream, think,
                     think_json, extra, remember):
        return {
            "base_url": base,
            "api_key": key,
            "model_name": model,
            "temperature": temp,
            "top_p": p,
            "max_tokens": tokens,
            "token_field": token_field_name,
            "presence_penalty": presence,
            "frequency_penalty": frequency,
            "seed": sd,
            "stop": stp,
            "timeout": tmo,
            "stream": is_stream,
            "thinking": think,
            "thinking_json": think_json,
            "extra_json": extra,
            "save_api_key": bool(remember),
        }

    def _save_api(*args):
        remember = bool(args[-1])
        typed_key = (args[1] or "").strip()
        effective_key = typed_key or api_key_state["value"]

        values = list(args)
        values[1] = effective_key
        payload = _api_payload(*values)
        try:
            llm_config.save_config(payload, save_api_key=remember)
            if typed_key:
                api_key_state["value"] = typed_key
            saved = "已保存（留空自动使用）" if api_key_state["value"] else "未保存"
            return (
                f"✅ API 配置已保存到 `{llm_config.CONFIG_PATH}`",
                gr.update(value="", placeholder=saved),
            )
        except Exception as exc:
            traceback.print_exc()
            return f"❌ 保存配置失败：{exc}", gr.update()

    def _clear_api():
        api_key_state["value"] = ""
        try:
            llm_config.clear_config()
            return (
                f"✅ 已删除 `{llm_config.CONFIG_PATH}`（界面上的值保持不变）",
                gr.update(value="", placeholder="未保存"),
            )
        except Exception as exc:
            return f"❌ 删除配置失败：{exc}", gr.update()

    def _llm_generate(base, key, model, temp, p, tokens, token_field_name,
                      presence, frequency,
                      sd, stp, tmo, is_stream, think, think_json, extra,
                      prompt, persona_text, story_text, as_user, remember,
                      progress=gr.Progress()):
        effective_key = (key or "").strip() or api_key_state["value"]
        if (key or "").strip():
            api_key_state["value"] = key.strip()

        try:
            llm_config.save_config(
                _api_payload(base, effective_key, model, temp, p, tokens,
                             token_field_name, presence, frequency, sd, stp, tmo,
                             is_stream, think, think_json, extra, remember),
                save_api_key=bool(remember),
            )
        except Exception as exc:
            print(f"[whole_voice] 自动保存 API 配置失败：{exc}")

        progress(0.1, desc="请求大语言模型……")
        try:
            segments_data = generate_segments_from_llm(
                base_url=base,
                api_key=effective_key,
                model_name=model,
                temperature=temp,
                top_p=p,
                max_tokens=tokens,
                token_field=token_field_name,
                presence_penalty=presence,
                frequency_penalty=frequency,
                seed=sd,
                stop=stp,
                timeout=tmo,
                stream=is_stream,
                thinking=think,
                thinking_json=think_json,
                extra_json=extra,
                system_prompt=prompt,
                persona=persona_text,
                story=story_text,
                system_as_user=as_user,
            )
        except Exception as exc:
            traceback.print_exc()
            return gr.update(), gr.update(), f"❌ {exc}"
        progress(1.0, desc="完成")
        rows = segments_to_rows(segments_data)
        return rows, segments_to_json(segments_data), f"✅ 已生成 {len(segments_data)} 段"

    def _refresh_characters():
        new_choices = [(entry["label"], entry["key"]) for entry in list_character_entries()]
        first = new_choices[0][1] if new_choices else None
        return gr.update(choices=new_choices, value=first), load_persona(first)

    def _on_chara_change(name):
        return load_persona(name)

    def _apply_template(name):
        config = API_TEMPLATES.get(name)
        if not config:
            return (gr.update(),) * 9
        return (
            gr.update(value=config["base_url"]),
            gr.update(value=config["model"]),
            gr.update(value=config["temperature"]),
            gr.update(value=config["top_p"]),
            gr.update(value=config["max_tokens"]),
            gr.update(value=config["token_field"]),
            gr.update(value=config["thinking"]),
            gr.update(value=config["thinking_json"]),
            gr.update(value=config["extra_json"]),
        )

    refresh_chara.click(_refresh_characters, [], [chara, persona])
    chara.change(_on_chara_change, [chara], [persona])
    persona_load_button.click(_on_chara_change, [chara], [persona])
    add_row.click(_add_row, [segments], [segments])
    del_row.click(_del_row, [segments], [segments])
    export_json.click(table_to_json, [segments], [segments_json, generate_status])
    import_json.click(json_to_table, [segments_json], [segments, generate_status])
    load_prompt_button.click(_load_prompt, [prompt_path], [system_prompt, llm_status])
    template.change(
        _apply_template,
        [template],
        [
            base_url,
            model_name,
            llm_temperature,
            llm_top_p,
            max_tokens,
            token_field,
            thinking,
            thinking_json,
            extra_json,
        ],
    )

    generate_button.click(
        _generate,
        [
            chara,
            segments,
            text_language,
            top_k,
            top_p,
            temperature,
            speed,
            sentence_gap,
            segment_gap,
            use_asr,
        ],
        [audio_output, generate_status],
    )

    save_config_button.click(
        _save_api,
        [
            base_url,
            api_key,
            model_name,
            llm_temperature,
            llm_top_p,
            max_tokens,
            token_field,
            presence_penalty,
            frequency_penalty,
            seed,
            stop,
            timeout,
            stream,
            thinking,
            thinking_json,
            extra_json,
            remember_key,
        ],
        [llm_status, api_key],
    )
    clear_config_button.click(_clear_api, [], [llm_status, api_key])

    llm_button.click(
        _llm_generate,
        [
            base_url,
            api_key,
            model_name,
            llm_temperature,
            llm_top_p,
            max_tokens,
            token_field,
            presence_penalty,
            frequency_penalty,
            seed,
            stop,
            timeout,
            stream,
            thinking,
            thinking_json,
            extra_json,
            system_prompt,
            persona,
            story,
            system_as_user,
            remember_key,
        ],
        [segments, segments_json, llm_status],
    )
