"""
VoxCPM2 配音工作台 · 单进程一体化服务。

把原来的三件套合并到同一个 FastAPI 进程里：
  1) dubbing_api.py     —— TTS / 缓存 / read_wav / health（/api/...）
  2) web/server.js      —— 静态托管 web/static + POST /resolve_name
  3) web/static/dubbing_workbench.html —— 前端页面（同源直连，无 CORS）

启动：
  python server.py
  # 浏览器打开 http://127.0.0.1:8770/           → 自动返回 dubbing_workbench.html
  # 或        http://127.0.0.1:8770/dubbing_workbench.html

环境变量：
  VOXCPM_PORT / DUBBING_PORT   端口，默认 8770（两者都识别，前者优先）
  VOXCPM_HOST                  监听地址，默认 127.0.0.1
  VOXCPM_RELOAD                1/true/yes 时 uvicorn --reload（会反复加载模型，慎用）
  VOXCPM_AUDIO_SEARCH_DIRS     额外搜索目录（Windows 分号、*nix 冒号分隔），供 /resolve_name 使用
  以及 dubbing_api.py 里原有的 VOXCPM_MODEL / VOXCPM_LOAD_DENOISER / VOXCPM_TTS_CACHE_DIR
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import Body, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from api import create_app

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT/ "static"
DEFAULT_PAGE = "webui.html"


class ResolveNameBody(BaseModel):
    filename: str = Field(..., description="仅文件名（不能含路径分隔符）")
    limit: int = Field(20, ge=1, le=200)
    per_dir_depth_limit: int = Field(6, ge=0, le=20)


def _search_dirs() -> list[Path]:
    """与 web/server.js 的 searchDirs() 对齐：常用目录 + VOXCPM_AUDIO_SEARCH_DIRS。"""
    home = Path.home()
    dirs: list[Path] = [
        home / "Downloads",
        home / "Desktop",
        home / "Documents",
        home / "Music",
        home / "Videos",
        ROOT.parent,
        ROOT,
        ROOT.parent / "assets",
    ]
    extra = (os.environ.get("VOXCPM_AUDIO_SEARCH_DIRS") or "").strip()
    if extra:
        sep = ";" if os.name == "nt" else ":"
        for d in extra.split(sep):
            d = d.strip()
            if d:
                dirs.append(Path(d))
    return dirs


def _resolve_filename(name: str, limit: int = 20, per_dir_depth_limit: int = 6) -> list[str]:
    target = name.lower()
    found: list[str] = []
    seen: set[str] = set()

    def walk(dir_path: Path, depth: int) -> None:
        if len(found) >= limit or depth > per_dir_depth_limit:
            return
        try:
            entries = list(os.scandir(dir_path))
        except OSError:
            return
        for e in entries:
            if len(found) >= limit:
                return
            try:
                if e.is_dir(follow_symlinks=False):
                    bn = e.name
                    if bn.startswith(".") or bn == "node_modules":
                        continue
                    walk(Path(e.path), depth + 1)
                elif e.is_file(follow_symlinks=False):
                    if e.name.lower() == target:
                        rp = str(Path(e.path).resolve())
                        if rp not in seen:
                            seen.add(rp)
                            found.append(rp)
            except OSError:
                continue

    for d in _search_dirs():
        if len(found) >= limit:
            break
        try:
            if d.is_dir():
                walk(d, 0)
        except OSError:
            continue
    return found


def _register_web_routes(app) -> None:
    """挂接 /resolve_name 与 web/static 静态托管。API 路由已由 dubbing_api.create_app 注册好。"""

    @app.post("/resolve_name")
    def resolve_name(body: dict = Body(...)):
        try:
            req = ResolveNameBody.model_validate(body)
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        name = req.filename.strip()
        if not name:
            raise HTTPException(status_code=400, detail="filename 为空")
        if "/" in name or "\\" in name:
            raise HTTPException(status_code=400, detail="filename 只能是文件名，不能含路径分隔符")
        found = _resolve_filename(name, limit=req.limit, per_dir_depth_limit=req.per_dir_depth_limit)
        return {"filename": name, "found": found}

    @app.get("/")
    def index():
        page = STATIC_DIR / DEFAULT_PAGE
        if not page.is_file():
            raise HTTPException(status_code=404, detail=f"缺少前端页面: {page}")
        return FileResponse(str(page), media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-store"})

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR)), name="static")
    else:
        print(f"[server] 警告：前端目录不存在 {STATIC_DIR}", file=sys.stderr)


app = create_app()
_register_web_routes(app)


def main() -> None:
    try:
        import uvicorn
    except ImportError:
        print("请先安装: pip install uvicorn", file=sys.stderr)
        raise SystemExit(1)

    port_raw = os.environ.get("VOXCPM_PORT") or os.environ.get("DUBBING_PORT") or "8770"
    try:
        port = int(port_raw)
    except ValueError:
        print(f"端口无效: {port_raw}", file=sys.stderr)
        raise SystemExit(2)
    host = os.environ.get("VOXCPM_HOST", "127.0.0.1")
    use_reload = (os.environ.get("VOXCPM_RELOAD") or "").lower() in ("1", "true", "yes")

    print(f"[server] VoxCPM2 一体化服务启动: http://{host}:{port}/", flush=True)
    print(f"[server] 前端页面: http://{host}:{port}/{DEFAULT_PAGE}", flush=True)
    print(f"[server] 静态目录: {STATIC_DIR}", flush=True)

    uvicorn.run(
        "server:app" if use_reload else app,
        host=host,
        port=port,
        reload=use_reload,
    )


if __name__ == "__main__":
    main()
