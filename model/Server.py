"""
配音工作台 — HTTP API（无静态页）。

由 server.py 启动；管理 OmniVoice / VoxCPM2 双后端模型的加载、切换与 TTS 推理。
"""

from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import re
import threading
import unicodedata
import uuid
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parent.parent


class RenderBody(BaseModel):
    text: str = Field(..., description="要合成的文本")
    reference_wav_path: str = Field(..., description="参考 wav 本机路径（克隆）；参与缓存 md5")
    role_name: str = Field("", description="角色名，仅用于日志/展示，不参与缓存 md5")
    cfg_value: float = Field(2.0)
    inference_timesteps: int = Field(10)
    tts_seed: int = Field(
        0,
        description="随机种子：参与磁盘缓存文件名 md5；相同台词/参考/cfg/步数下不同种子为不同缓存项",
    )


class CacheDigestItem(BaseModel):
    reference_wav_path: str = ""
    text: str = ""
    cfg_value: float = Field(2.0)
    inference_timesteps: int = Field(10)
    tts_seed: int = Field(0)


class CacheDigestBatch(BaseModel):
    items: list[CacheDigestItem] = Field(default_factory=list)


class WorkbenchReadWavBody(BaseModel):
    path: str = Field(..., description="服务器本机 .wav 路径")


class ModelSelectBody(BaseModel):
    id: str = Field(..., description="model/ 下的模型目录名")


_HEX32 = re.compile(r"^[a-fA-F0-9]{32}$")


def _tts_cache_dir() -> Path:
    raw = os.environ.get("VOXCPM_TTS_CACHE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return ROOT / "tts_cache"


def _norm_txt(s: str) -> str:
    t = (s or "").strip()
    t = unicodedata.normalize("NFC", t)
    return t


def _norm_ref_path(p: str) -> str:
    raw = (p or "").strip()
    if not raw:
        return ""
    try:
        ap = Path(raw).expanduser().resolve(strict=False)
    except OSError:
        ap = Path(raw).expanduser()
    s = str(ap)
    if os.name == "nt":
        s = s.replace("/", "\\").lower()
    else:
        s = os.path.normpath(s)
    return s


def _tts_cache_payload(
    model_id: str,
    reference_wav_path: str,
    text: str,
    cfg_value: float,
    inference_timesteps: int,
    tts_seed: int,
) -> bytes:
    payload = {
        "model": model_id,
        "ref": _norm_ref_path(reference_wav_path),
        "text": _norm_txt(text),
        "cfg": round(float(cfg_value), 6),
        "steps": int(inference_timesteps),
        "seed": int(tts_seed),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _tts_cache_digest(
    model_id: str,
    reference_wav_path: str,
    text: str,
    cfg_value: float,
    inference_timesteps: int,
    tts_seed: int,
) -> str:
    return hashlib.md5(
        _tts_cache_payload(model_id, reference_wav_path, text, cfg_value, inference_timesteps, tts_seed)
    ).hexdigest()


def _tts_cache_wav_path(
    model_id: str,
    reference_wav_path: str,
    text: str,
    cfg_value: float,
    inference_timesteps: int,
    tts_seed: int,
) -> Path:
    name = _tts_cache_digest(model_id, reference_wav_path, text, cfg_value, inference_timesteps, tts_seed) + ".wav"
    return _tts_cache_dir() / name


def _discover_local_models() -> dict[str, Path]:
    model_root = ROOT / "model"
    if not model_root.is_dir():
        return {}
    out: dict[str, Path] = {}
    for path in sorted(model_root.iterdir(), key=lambda p: p.name.lower()):
        if path.is_dir() and (path / "config.json").is_file():
            out[path.name] = path
    return out


def _model_type_for_path(path: Path) -> str:
    try:
        data = json.loads((path / "config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    return str(data.get("model_type") or "").lower()


def _resolve_model_id(cli_model: str) -> str:
    raw = (cli_model or "").strip()
    local_models = _discover_local_models()
    if raw in local_models:
        return str(local_models[raw])
    if raw == "openbmb/VoxCPM2" and "VoxCPM2" in local_models:
        return str(local_models["VoxCPM2"])
    if raw:
        p = Path(raw).expanduser()
        if p.is_dir() and (p / "config.json").is_file():
            return str(p)
        return raw
    if "VoxCPM2" in local_models:
        return str(local_models["VoxCPM2"])
    if local_models:
        return str(next(iter(local_models.values())))
    return "openbmb/VoxCPM2"


def _check_torch_numpy_bridge() -> None:
    import torch

    _ = torch.zeros(1, dtype=torch.float32).cpu().numpy()


def create_app():
    _check_torch_numpy_bridge()

    import torch
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, Response

    from model.OmniVoice import OmniVoiceService
    from model.VoxCPM2 import VoxCPM2Service

    load_denoiser = os.environ.get("VOXCPM_LOAD_DENOISER", "").lower() in ("1", "true", "yes")
    prefer_device = "cuda" if torch.cuda.is_available() else "cpu"
    model_lock = threading.Lock()
    state = {"id": "", "name": "", "path": "", "model": None, "sample_rate": 48000}

    if prefer_device == "cuda":
        print("[api] 检测到 CUDA，优先使用 GPU 推理", flush=True)
    else:
        print("[api] 未检测到 CUDA，使用 CPU 推理", flush=True)

    def _model_options() -> dict[str, Path]:
        return _discover_local_models()

    def _model_id_for_path(path: str) -> str:
        try:
            resolved = Path(path).expanduser().resolve(strict=False)
        except OSError:
            resolved = Path(path).expanduser()
        for model_key, model_path in _model_options().items():
            try:
                if model_path.resolve(strict=False) == resolved:
                    return model_key
            except OSError:
                pass
        return resolved.name or str(resolved)

    def _load_model(model_key: str):
        options = _model_options()
        if model_key not in options:
            raise HTTPException(status_code=404, detail=f"未找到模型: {model_key}")
        model_path_obj = options[model_key]
        model_path = str(model_path_obj)
        model_type = _model_type_for_path(model_path_obj)
        print(f"[api] 加载模型: {model_path} (type={model_type}) …", flush=True)

        if model_type == "omnivoice":
            try:
                device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                service = OmniVoiceService(model_path, device_map=device_map, dtype=dtype)
            except ImportError as e:
                raise HTTPException(status_code=500, detail="OmniVoice 模型需要先安装依赖：pip install omnivoice") from e
        else:
            try:
                service = VoxCPM2Service(model_path, load_denoiser=load_denoiser, device=prefer_device)
            except ImportError as e:
                raise HTTPException(status_code=500, detail="VoxCPM2 模型需要先安装依赖：pip install voxcpm") from e

        print(f"[api] 模型就绪: {model_key} ({service.backend})，采样率 {service.sample_rate} Hz", flush=True)
        return service, service.sample_rate, model_path

    def _model_state_payload() -> dict:
        models = []
        active = str(state["id"])
        for model_key, model_path in _model_options().items():
            models.append(
                {
                    "id": model_key,
                    "name": model_key,
                    "path": str(model_path),
                    "active": model_key == active,
                }
            )
        return {
            "ok": True,
            "models": models,
            "active": active,
            "model": active,
            "model_path": state["path"],
            "sample_rate": state["sample_rate"],
        }

    def _release_model(old_model) -> None:
        if old_model is not None and hasattr(old_model, "release"):
            old_model.release()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _select_model(model_key: str) -> dict:
        with model_lock:
            if state["model"] is not None and state["id"] == model_key:
                return _model_state_payload()
            loaded_model, loaded_sr, model_path = _load_model(model_key)
            old_model = state["model"]
            state.update({"id": model_key, "name": model_key, "path": model_path, "model": loaded_model, "sample_rate": loaded_sr})
            if old_model is not None:
                _release_model(old_model)
            return _model_state_payload()

    initial_model = _model_id_for_path(_resolve_model_id(os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2")))
    _select_model(initial_model)

    _cd = _tts_cache_dir()
    print(f"[api] 就绪，采样率 {state['sample_rate']} Hz", flush=True)
    try:
        _cd.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[api] 无法创建缓存目录 {_cd}: {e}", flush=True)
    print(
        f"[api] TTS 缓存目录（绝对路径）: {_cd.resolve()}",
        flush=True,
    )
    print(
        "[api] 文件名: <md5(模型+参考路径+台词+cfg+步数+tts_seed)>.wav；环境变量 VOXCPM_TTS_CACHE_DIR 可改为例如 D:\\cache 或 /tmp",
        flush=True,
    )

    app = FastAPI(title="VoxCPM2 Dubbing API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Sample-Rate", "X-Tts-Cache", "X-Tts-Cache-Digest"],
    )

    @app.post("/api/tts/render")
    def api_tts_render(body: dict = Body(...)):
        try:
            req = RenderBody.model_validate(body)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())

        text = _norm_txt(req.text)
        if not text:
            raise HTTPException(status_code=400, detail="text 为空")

        ref = Path(req.reference_wav_path).expanduser()
        if not ref.is_file():
            raise HTTPException(status_code=400, detail=f"参考音频不存在: {ref}")

        ref_s = str(ref.resolve())
        with model_lock:
            model = state["model"]
            model_id = str(state["id"])
            sr = int(state["sample_rate"])
            if model is None:
                raise HTTPException(status_code=503, detail="模型尚未加载")
            cache_path = _tts_cache_wav_path(model_id, ref_s, text, req.cfg_value, req.inference_timesteps, req.tts_seed)
            digest = cache_path.stem

            if cache_path.is_file():
                try:
                    import soundfile as sf

                    info = sf.info(str(cache_path))
                    sr_file = int(info.samplerate)
                    data = cache_path.read_bytes()
                    if len(data) < 100:
                        raise ValueError("缓存 wav 过小，可能损坏")
                    if sr_file != sr:
                        print(
                            f"[api] TTS 缓存命中但采样率与当前模型声明不一致（文件 {sr_file} Hz，模型 {sr} Hz），仍返回缓存文件",
                            flush=True,
                        )
                    print(f"[api] TTS 缓存命中: {cache_path.name} digest={digest[:16]}…", flush=True)
                    return Response(
                        content=data,
                        media_type="audio/wav",
                        headers={
                            "X-Sample-Rate": str(sr_file),
                            "X-Tts-Cache": "hit",
                            "X-Tts-Cache-Digest": digest,
                            "Cache-Control": "no-store",
                        },
                    )
                except Exception as e:
                    print(f"[api] TTS 缓存读失败，将重算: {cache_path} — {type(e).__name__}: {e}", flush=True)
                try:
                    cache_path.unlink(missing_ok=True)
                except OSError:
                    pass
            else:
                print(
                    f"[api] TTS 缓存未命中: model={model_id} 无文件 digest={digest[:16]}… path={cache_path} seed={req.tts_seed}",
                    flush=True,
                )

            print(f"[api] TTS 推理开始 — model={model_id} backend={model.backend} sample_rate={sr}", flush=True)
            print(f"[api]   text: {text[:150]}{'…' if len(text) > 150 else ''}", flush=True)
            print(f"[api]   reference_wav_path: {ref_s}", flush=True)
            print(f"[api]   cfg_value: {req.cfg_value}", flush=True)
            print(f"[api]   inference_timesteps: {req.inference_timesteps}", flush=True)
            print(f"[api]   tts_seed: {req.tts_seed}", flush=True)
            print(f"[api]   role_name: {req.role_name or '(空)'}", flush=True)

            if model.backend == "omnivoice":
                wav = model.generate(
                    text=text,
                    reference_wav_path=ref_s,
                    cfg_value=req.cfg_value,
                    inference_timesteps=req.inference_timesteps,
                    seed=req.tts_seed,
                )
            else:
                wav = model.generate(
                    text=text,
                    reference_wav_path=ref_s,
                    cfg_value=req.cfg_value,
                    inference_timesteps=req.inference_timesteps,
                    seed=req.tts_seed,
                )

            arr = np.asarray(wav, dtype=np.float32).reshape(-1)

            import soundfile as sf

            tmp = None
            try:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = cache_path.parent / f".{digest}.{uuid.uuid4().hex}.tmp.wav"
                sf.write(str(tmp), arr, sr, format="WAV", subtype="PCM_16")
                os.replace(str(tmp), str(cache_path))
                tmp = None
                data = cache_path.read_bytes()
                print(f"[api] TTS 已写入缓存: {cache_path.name} digest={digest[:16]}…", flush=True)
            except OSError as e:
                print(f"[api] 写入 TTS 缓存失败（回退内存编码）: {cache_path} — {e}", flush=True)
                try:
                    if tmp is not None and tmp.is_file():
                        tmp.unlink(missing_ok=True)
                except OSError:
                    pass
                bio = io.BytesIO()
                sf.write(bio, arr, sr, format="WAV", subtype="PCM_16")
                data = bio.getvalue()
            return Response(
                content=data,
                media_type="audio/wav",
                headers={
                    "X-Sample-Rate": str(sr),
                    "X-Tts-Cache": "miss",
                    "X-Tts-Cache-Digest": digest,
                    "Cache-Control": "no-store",
                },
            )

    @app.post("/api/tts/cache_digests")
    def api_tts_cache_digests(body: dict = Body(...)):
        try:
            batch = CacheDigestBatch.model_validate(body)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        with model_lock:
            model_id = str(state["id"])
            out: list[dict] = []
            for it in batch.items:
                ref = it.reference_wav_path or ""
                text = _norm_txt(it.text or "")
                d = _tts_cache_digest(model_id, ref, text, it.cfg_value, it.inference_timesteps, it.tts_seed)
                out.append({"digest": d, "relativePath": f"tts_cache/{d}.wav"})
            return {"digests": out}

    @app.get("/api/tts/cache_wav/{digest}")
    def api_tts_cache_wav(digest: str):
        if not _HEX32.match(digest or ""):
            raise HTTPException(status_code=400, detail="digest 须为 32 位十六进制")
        d = digest.lower()
        path = _tts_cache_dir() / f"{d}.wav"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="无此缓存文件，请先对该台词调用一次生成")
        return FileResponse(str(path), media_type="audio/wav", filename=f"{d}.wav")

    @app.post("/api/workbench/read_wav")
    def api_workbench_read_wav(body: dict = Body(...)):
        try:
            req = WorkbenchReadWavBody.model_validate(body)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        p = Path(req.path).expanduser()
        try:
            p = p.resolve()
        except OSError as e:
            raise HTTPException(status_code=400, detail=f"路径无效: {e}") from e
        if not p.is_file():
            raise HTTPException(status_code=400, detail=f"文件不存在: {p}")
        if p.suffix.lower() != ".wav":
            raise HTTPException(status_code=400, detail="仅支持 .wav 文件")
        return FileResponse(str(p), media_type="audio/wav", filename=p.name, headers={"Cache-Control": "no-store"})

    @app.get("/api/models")
    def api_models():
        with model_lock:
            return _model_state_payload()

    @app.post("/api/models/select")
    def api_models_select(body: dict = Body(...)):
        try:
            req = ModelSelectBody.model_validate(body)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        return _select_model(req.id)

    @app.get("/api/health")
    def health():
        with model_lock:
            return _model_state_payload()

    return app
