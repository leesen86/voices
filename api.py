"""
VoxCPM2 配音工作台 — 仅 HTTP API（无静态页）。

由 dubbing_server.py 启动；调试 HTML 时请用 serve_dubbing_page.py 单独起静态服务，
这样改页面只需刷新，不必重启本进程（避免重复加载模型）。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import unicodedata
import uuid
from pathlib import Path

import numpy as _np  # noqa: F401
from pydantic import BaseModel, Field, ValidationError

ROOT = Path(__file__).resolve().parent


class RenderBody(BaseModel):
    text: str = Field(..., description="要合成的文本")
    reference_wav_path: str = Field(..., description="参考 wav 本机路径（克隆）；参与缓存 md5")
    role_name: str = Field("", description="角色名，仅用于日志/展示，不参与缓存 md5")
    cfg_value: float = Field(2.0)
    inference_timesteps: int = Field(10)


class CacheDigestItem(BaseModel):
    reference_wav_path: str = ""
    text: str = ""
    cfg_value: float = Field(2.0)
    inference_timesteps: int = Field(10)


class CacheDigestBatch(BaseModel):
    items: list[CacheDigestItem] = Field(default_factory=list)


class WorkbenchReadWavBody(BaseModel):
    """工作台从本机路径读取整段 WAV（与 reference_wav_path 相同信任模型：仅本地联调服务）。"""

    path: str = Field(..., description="服务器本机 .wav 路径")


_HEX32 = re.compile(r"^[a-fA-F0-9]{32}$")


def _tts_cache_dir() -> Path:
    """磁盘缓存目录。未设置 VOXCPM_TTS_CACHE_DIR 时默认用项目根下 tts_cache/（Windows 也可直接看到文件）。"""
    raw = os.environ.get("VOXCPM_TTS_CACHE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return ROOT / "tts_cache"


def _norm_txt(s: str) -> str:
    """统一空白与 Unicode 规范化，避免同一台词因 NFC/NFD 或不可见字符导致 md5 不一致。"""
    t = (s or "").strip()
    t = unicodedata.normalize("NFC", t)
    return t


def _norm_ref_path(p: str) -> str:
    """把参考音频路径规范化成稳定 key：

    - ``~`` 展开、``resolve()`` 取绝对路径，消除相对路径/软链差异。
    - Windows 的 NTFS 不区分大小写，且正反斜杠混用常见，所以统一成小写 + ``\\``。
    - 其它系统保持原样大小写，只做 ``os.path.normpath``。

    目的：同一个物理 wav 文件不管用户怎么写，都映射到同一个 md5。
    """
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
    reference_wav_path: str,
    text: str,
    cfg_value: float,
    inference_timesteps: int,
) -> bytes:
    payload = {
        "ref": _norm_ref_path(reference_wav_path),
        "text": _norm_txt(text),
        "cfg": round(float(cfg_value), 6),
        "steps": int(inference_timesteps),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _tts_cache_digest(
    reference_wav_path: str,
    text: str,
    cfg_value: float,
    inference_timesteps: int,
) -> str:
    return hashlib.md5(
        _tts_cache_payload(reference_wav_path, text, cfg_value, inference_timesteps)
    ).hexdigest()


def _tts_cache_wav_path(
    reference_wav_path: str,
    text: str,
    cfg_value: float,
    inference_timesteps: int,
) -> Path:
    name = _tts_cache_digest(reference_wav_path, text, cfg_value, inference_timesteps) + ".wav"
    return _tts_cache_dir() / name


def _check_torch_numpy_bridge() -> None:
    import torch

    _ = torch.zeros(1, dtype=torch.float32).cpu().numpy()


def _resolve_model_id(cli_model: str) -> str:
    model_id = cli_model
    local_model = ROOT / "model"
    if model_id == "openbmb/VoxCPM2" and local_model.is_dir() and (local_model / "config.json").is_file():
        model_id = str(local_model)
    return model_id


def create_app():
    _check_torch_numpy_bridge()

    import numpy as np
    import soundfile as sf
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, Response
    from voxcpm import VoxCPM

    model_id = _resolve_model_id(os.environ.get("VOXCPM_MODEL", "openbmb/VoxCPM2"))
    load_denoiser = os.environ.get("VOXCPM_LOAD_DENOISER", "").lower() in ("1", "true", "yes")
    print(f"[api] 加载模型: {model_id} …", flush=True)
    model = VoxCPM.from_pretrained(model_id, load_denoiser=load_denoiser)
    sr = int(model.tts_model.sample_rate)
    _cd = _tts_cache_dir()
    print(f"[api] 就绪，采样率 {sr} Hz", flush=True)
    try:
        _cd.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[api] 无法创建缓存目录 {_cd}: {e}", flush=True)
    print(
        f"[api] TTS 缓存目录（绝对路径）: {_cd.resolve()}",
        flush=True,
    )
    print(
        "[api] 文件名: <md5(角色路径+台词+cfg+步数)>.wav；环境变量 VOXCPM_TTS_CACHE_DIR 可改为例如 D:\\\\cache 或 /tmp",
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
        cache_path = _tts_cache_wav_path(ref_s, text, req.cfg_value, req.inference_timesteps)
        digest = cache_path.stem

        if cache_path.is_file():
            try:
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
                f"[api] TTS 缓存未命中: 无文件 digest={digest[:16]}… path={cache_path}",
                flush=True,
            )

        wav = model.generate(
            text=text,
            reference_wav_path=ref_s,
            cfg_value=req.cfg_value,
            inference_timesteps=req.inference_timesteps,
        )
        arr = np.asarray(wav, dtype=np.float32).reshape(-1)

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
        """与磁盘缓存文件名一致的 md5 列表（不含音频体），供工作台导出 JSON。"""
        try:
            batch = CacheDigestBatch.model_validate(body)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        out: list[dict] = []
        for it in batch.items:
            ref = it.reference_wav_path or ""
            text = _norm_txt(it.text or "")
            d = _tts_cache_digest(ref, text, it.cfg_value, it.inference_timesteps)
            out.append({"digest": d, "relativePath": f"tts_cache/{d}.wav"})
        return {"digests": out}

    @app.get("/api/tts/cache_wav/{digest}")
    def api_tts_cache_wav(digest: str):
        """按 md5 读取服务端已缓存的整段 WAV（须已生成过 TTS）。"""
        if not _HEX32.match(digest or ""):
            raise HTTPException(status_code=400, detail="digest 须为 32 位十六进制")
        d = digest.lower()
        path = _tts_cache_dir() / f"{d}.wav"
        if not path.is_file():
            raise HTTPException(status_code=404, detail="无此缓存文件，请先对该台词调用一次生成")
        return FileResponse(str(path), media_type="audio/wav", filename=f"{d}.wav")

    @app.post("/api/workbench/read_wav")
    def api_workbench_read_wav(body: dict = Body(...)):
        """按本机路径返回 WAV 字节，供工作台背景音轨等浏览器解码（路径须存在且为 .wav）。"""
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

    @app.get("/api/health")
    def health():
        return {"ok": True, "sample_rate": sr, "model": model_id}

    return app


app = create_app()
