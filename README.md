# VoxCPM2 配音工作台 · 一体化服务

> 单进程把 **模型推理 API** + **前端页面** + **本机文件定位** 全挂在一个 FastAPI 上，默认监听 `http://127.0.0.1:8770/`。

---

## 快速开始


## 下载
```powershell
# 1) 下载模型
pip install -U "huggingface_hub[cli]"
huggingface-cli download openbmb/VoxCPM2 --local-dir ./model

# 2) 安装依赖
pip install -r requirements.txt

# 3) 启动
python server.py

# 4) 浏览器打开
#    http://127.0.0.1:8770/
#    → 自动呈现 static/webui.html
```

看到类似这样就表示就绪：

```
[api] 加载模型: ...\model …
[api] 就绪，采样率 16000 Hz
[api] TTS 缓存目录（绝对路径）: D:\AI\VoxCPM2\tts_cache
[api] 文件名: <md5(角色路径+台词+cfg+步数)>.wav；环境变量 VOXCPM_TTS_CACHE_DIR 可改为例如 D:\\cache 或 /tmp
[server] VoxCPM2 一体化服务启动: http://127.0.0.1:8770/
[server] 前端页面: http://127.0.0.1:8770/webui.html
```

---

## 项目结构

```
VoxCPM2/
├─ server.py           ← 入口：FastAPI 一体化服务（API + 静态 + /resolve_name）
├─ api.py              ← TTS 路由（/api/tts/*、/api/workbench/*、/api/health）
├─ requirements.txt    ← Python 依赖
├─ static/             ← 被 server.py 的 StaticFiles 挂载
│  └─ webui.html       ← 前端工作台（打开 "/" 默认返回此页）
├─ model/              ← 本地 VoxCPM2 模型目录（含 config.json 时自动走本地）
└─ tts_cache/          ← 磁盘缓存，文件名 = <md5>.wav
```

---

## 环境变量

| 变量 | 默认 | 说明 |
|---|---|---|
| `VOXCPM_PORT` | `8770` | 监听端口 |
| `VOXCPM_HOST` | `127.0.0.1` | 监听地址；改 `0.0.0.0` 后同局域网可访问 |
| `VOXCPM_RELOAD` | 空 | `1`/`true`/`yes` 时开启 uvicorn `--reload`（改代码即重载，但**会反复加载模型**，慎用） |
| `VOXCPM_MODEL` | `openbmb/VoxCPM2` | HF 仓库 ID 或本地目录。默认若 `./model/config.json` 存在，会自动切到本地目录 |
| `VOXCPM_LOAD_DENOISER` | 空 | `1`/`true`/`yes` 时一并加载 denoiser |
| `VOXCPM_TTS_CACHE_DIR` | `./tts_cache` | 缓存目录。例如换到固态盘 `D:\cache` |
| `VOXCPM_AUDIO_SEARCH_DIRS` | 空 | 供 `/resolve_name` 使用的额外搜索目录；Windows 用 `;` 分隔，*nix 用 `:` |

PowerShell 设置示例：

```powershell
$env:VOXCPM_PORT = "9000"
$env:VOXCPM_TTS_CACHE_DIR = "D:\cache\voxcpm"
$env:VOXCPM_AUDIO_SEARCH_DIRS = "D:\voice_refs;E:\bgm"
python server.py
```

---

## API 一览

同源调用（浏览器打开 `http://127.0.0.1:8770/webui.html` 时所有 `fetch` 全走同源，无 CORS 问题），也支持 `Access-Control-Allow-Origin: *` 跨源。

### `GET /api/health`

健康检查：

```json
{ "ok": true, "sample_rate": 16000, "model": "...\\model" }
```

### `POST /api/tts/render` —— 合成单句

请求体：

```json
{
  "text": "(语气平缓)这是一个示例。",
  "reference_wav_path": "C:\\Users\\you\\Downloads\\spk.wav",
  "role_name": "旁白",
  "cfg_value": 2.0,
  "inference_timesteps": 10
}
```

- 响应：`Content-Type: audio/wav`，直接返回 WAV 字节。
- 响应头：
  - `X-Sample-Rate`：采样率。
  - `X-Tts-Cache`：`hit` / `miss`。
  - `X-Tts-Cache-Digest`：缓存文件 md5（= 磁盘上 `tts_cache/<digest>.wav`）。
- **缓存 key**：`md5(规范化参考路径 + 台词 + cfg + 步数)`。
  - 路径规范化：`expanduser + resolve`，Windows 下额外统一为小写 + 反斜杠。
  - `role_name` **不**参与 md5，仅用于日志/展示。
  - 换参考音频路径 = 自动 miss 重算，避免错用旧缓存。

### `POST /api/tts/cache_digests` —— 批量预测 digest

用于导出「台词清单 → 文件名」映射，不触发推理。

```json
{
  "items": [
    { "reference_wav_path": "C:\\spk.wav", "text": "你好", "cfg_value": 2.0, "inference_timesteps": 10 }
  ]
}
```

响应：

```json
{ "digests": [ { "digest": "abc...32位", "relativePath": "tts_cache/abc...32位.wav" } ] }
```

### `GET /api/tts/cache_wav/{digest}`

按 md5 取已缓存的整段 WAV；未命中返回 404（说明该台词尚未生成过）。`digest` 必须是 32 位十六进制。

### `POST /api/workbench/read_wav`

把本机 `.wav` 原样回传，给前端背景音轨解码用：

```json
{ "path": "C:\\music\\bgm.wav" }
```

### `POST /resolve_name`

按**文件名**在常用目录中递归搜索本机绝对路径（前端「选择音轨」后用它把文件名还原成完整路径）：

```json
{ "filename": "spk_1765183119.wav", "limit": 20 }
```

响应：

```json
{ "filename": "spk_1765183119.wav", "found": ["C:\\Users\\you\\Downloads\\spk_1765183119.wav"] }
```

默认搜索目录（按顺序）：

- `~/Downloads`、`~/Desktop`、`~/Documents`、`~/Music`、`~/Videos`
- 项目根目录的父目录
- 项目根目录
- 项目根目录下的 `assets/`
- `VOXCPM_AUDIO_SEARCH_DIRS` 追加目录

`filename` 不允许含 `/` 或 `\`（只接受纯文件名，防路径注入）。

---

## 缓存目录 `tts_cache/`

- 每条成功生成的台词都会落盘：`tts_cache/<md5>.wav`，下次相同参数（路径 / 台词 / cfg / 步数）秒出。
- 想重新生成某一条：删掉对应的 `<md5>.wav` 即可；或清空整个目录：

```powershell
Remove-Item -Path .\tts_cache\*.wav -Force
```

---

## 前端页面要点

- 入口：`static/webui.html`，由 `server.py` 通过 `StaticFiles` 挂载。
- 改完 HTML 直接**刷新浏览器**即可，不必重启 Python 进程（静态服务对 `Cache-Control: no-store` 友好）。
- 页面里 `API` 常量默认 `http://127.0.0.1:8770`，也可以用 `?api=http://...` 查询参数显式指定（方便把页面托管在其他地方、API 跑在另一台机器的场景）。

---

## 常见问题

### 启动时报 `RuntimeError: Numpy is not available` / `_multiarray_umath`

典型 Windows DLL 加载顺序问题，代码里已经在模型加载前做了一次 `torch.zeros(1).numpy()` 探测。修复：

```powershell
pip uninstall numpy -y
conda install -y numpy -c conda-forge
# 仍失败则装/修 Microsoft Visual C++ Redistributable
```

### `pip install torch` 装成了 CPU 版，想用 CUDA

先按官方命令装 torch，再装其它依赖：

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 端口被占

```powershell
$env:VOXCPM_PORT = "9001"; python server.py
```

### 局域网其它机器访问

```powershell
$env:VOXCPM_HOST = "0.0.0.0"; python server.py
# 然后用本机 IPv4 地址 + 端口访问；注意防火墙放行
```

---

## 停止

终端里 `Ctrl+C`。
