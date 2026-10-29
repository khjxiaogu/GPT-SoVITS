# llm_config.py
"""LLM API 配置的本地保存与读取。

安全策略：
- 除 API Key 外的参数以明文 JSON 存到项目内 `configs/llm_config.json`，方便迁移；
- API Key 使用 Windows DPAPI（CryptProtectData）加密后再落盘，只有当前
  Windows 用户能解密；换机器/换用户后解密失败会返回空字符串，不会报错；
- 写入采用临时文件 + 原子替换，并尽量把文件权限收紧到当前用户；
- 非 Windows 平台不保存 API Key（只保存其余参数）。
"""

import base64
import ctypes
import json
import os

CONFIG_DIR = "configs"
CONFIG_PATH = os.path.join(CONFIG_DIR, "llm_config.json")
KEY_FIELD = "api_key_enc"

# 允许持久化的字段（白名单，避免把界面上的临时内容写进去）
CONFIG_FIELDS = [
    "base_url",
    "model_name",
    "token_field",
    "timeout",
    "temperature",
    "top_p",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "seed",
    "stop",
    "stream",
    "thinking",
    "thinking_json",
    "extra_json",
    "system_as_user",
    "save_api_key",
]

_ENTROPY = b"gpt-sovits-whole-voice-llm-config"


class _DataBlob(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.c_uint32),
        ("pbData", ctypes.POINTER(ctypes.c_char)),
    ]


def _make_blob(data):
    buffer = ctypes.create_string_buffer(data, len(data))
    return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_char))), buffer


if os.name == "nt":
    _crypt32 = ctypes.windll.crypt32
    _kernel32 = ctypes.windll.kernel32
    _crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.c_wchar_p,
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(_DataBlob),
    ]
    _crypt32.CryptProtectData.restype = ctypes.c_int
    _crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DataBlob),
        ctypes.POINTER(ctypes.c_wchar_p),
        ctypes.POINTER(_DataBlob),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.POINTER(_DataBlob),
    ]
    _crypt32.CryptUnprotectData.restype = ctypes.c_int
    _kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    _kernel32.LocalFree.restype = ctypes.c_void_p


def _protect(text):
    """用当前用户的 DPAPI 加密字符串，返回 base64 文本。"""
    if os.name != "nt":
        raise OSError("当前平台不支持 DPAPI")

    data = text.encode("utf-8")
    blob_in, buffer_in = _make_blob(data)
    entropy_blob, buffer_entropy = _make_blob(_ENTROPY)
    blob_out = _DataBlob()

    ok = _crypt32.CryptProtectData(
        ctypes.byref(blob_in),
        None,
        ctypes.byref(entropy_blob),
        None,
        None,
        0x1,  # CRYPTPROTECT_UI_FORBIDDEN
        ctypes.byref(blob_out),
    )
    if not ok:
        raise ctypes.WinError(ctypes.get_last_error())

    try:
        cipher = ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        _kernel32.LocalFree(blob_out.pbData)

    # 防止 buffer 被优化掉
    del buffer_in, buffer_entropy
    return base64.b64encode(cipher).decode("ascii")


def _unprotect(token):
    """解密 DPAPI 文本；失败返回 None。"""
    if os.name != "nt" or not token:
        return None

    try:
        cipher = base64.b64decode(token)
        blob_in, buffer_in = _make_blob(cipher)
        entropy_blob, buffer_entropy = _make_blob(_ENTROPY)
        blob_out = _DataBlob()

        ok = _crypt32.CryptUnprotectData(
            ctypes.byref(blob_in),
            None,
            ctypes.byref(entropy_blob),
            None,
            None,
            0x1,
            ctypes.byref(blob_out),
        )
        if not ok:
            return None

        try:
            data = ctypes.string_at(blob_out.pbData, blob_out.cbData)
        finally:
            _kernel32.LocalFree(blob_out.pbData)

        del buffer_in, buffer_entropy
        return data.decode("utf-8")
    except Exception:
        return None


def load_config():
    """读取配置；文件不存在、损坏或密钥无法解密时返回可用的部分。"""
    if not os.path.isfile(CONFIG_PATH):
        return {}

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as file:
            saved = json.load(file)
    except Exception:
        return {}
    if not isinstance(saved, dict):
        return {}

    config = {key: saved[key] for key in CONFIG_FIELDS if key in saved}

    encrypted_key = saved.get(KEY_FIELD)
    if encrypted_key:
        api_key = _unprotect(encrypted_key)
        if api_key:
            config["api_key"] = api_key
        else:
            print("[llm_config] 已保存的 API Key 无法解密（可能换了用户或机器），已忽略")

    return config


def save_config(config, save_api_key=True):
    """保存配置。

    save_api_key 为 False 时不保存（并清除旧的）API Key；
    为 True 且本次没有传入新 Key 时，保留文件里已有的加密 Key。
    """
    payload = {key: config.get(key) for key in CONFIG_FIELDS if key in config}
    payload["save_api_key"] = bool(save_api_key)

    api_key = (config.get("api_key") or "").strip()
    if save_api_key and api_key:
        try:
            payload[KEY_FIELD] = _protect(api_key)
        except Exception as exc:
            print(f"[llm_config] API Key 加密失败，本次不保存 Key：{exc}")
    elif save_api_key:
        # 界面不回显 Key，留空保存时沿用已有的加密 Key
        existing = {}
        if os.path.isfile(CONFIG_PATH):
            try:
                with open(CONFIG_PATH, "r", encoding="utf-8") as file:
                    existing = json.load(file)
            except Exception:
                existing = {}
        if isinstance(existing, dict) and existing.get(KEY_FIELD):
            payload[KEY_FIELD] = existing[KEY_FIELD]

    os.makedirs(CONFIG_DIR, exist_ok=True)
    temp_path = CONFIG_PATH + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    try:
        os.chmod(temp_path, 0o600)
    except Exception:
        pass

    os.replace(temp_path, CONFIG_PATH)
    return payload


def clear_config():
    """删除已保存的配置（不影响界面当前值）。"""
    for path in (CONFIG_PATH, CONFIG_PATH + ".tmp"):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except Exception as exc:
            print(f"[llm_config] 删除配置失败：{exc}")
