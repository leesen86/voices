# 环境安装说明（GPU / PyTorch）

本文整理在 **Windows + PowerShell** 下，用 **pip** 安装带 **CUDA 12.8（cu128）** 的 PyTorch 及相关命令，与当前常见驱动（`nvidia-smi` 显示 CUDA 12.8）对齐。

## 前置条件

- 已安装 NVIDIA 驱动，且 `nvidia-smi` 能正常显示 GPU。
- Python 3.10+（示例为 Anaconda 自带的 `python.exe`）。

## 本项目依赖（含 voxcpm）

`api.py` 会 `import voxcpm`。若出现 **`ModuleNotFoundError: No module named 'voxcpm'`**，说明尚未安装依赖。在仓库根目录、用运行 `server.py` 的**同一个** Python 执行：

```powershell
cd D:\project\voices
python -m pip install -r requirements.txt
```

仅快速补齐 **voxcpm** 时：

```powershell
python -m pip install voxcpm
```

说明：`voxcpm` 会拉取较多依赖（如 `gradio`、`datasets`、`funasr` 等），首次安装可能较久。若已按下文装好 **CUDA 版** `torch`，再执行 `pip install -r requirements.txt` 时若发现 `torch` 被 PyPI 换成 **`+cpu`**，请再执行一次下文「一键安装」中的 `torch` 命令以恢复 **cu128**。

## 一键安装（推荐）

使用与 [PyTorch 官网](https://pytorch.org/get-started/locally/) 一致的 **cu128** 索引：

```powershell
C:\ProgramData\anaconda3\python.exe -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

若你的 Python 不在上述路径，请改成你的解释器，例如：

```powershell
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 先卸载再安装（可选）

当需要彻底替换旧版（例如从 `+cpu` 换成 `+cu128`）时：

```powershell
C:\ProgramData\anaconda3\python.exe -m pip uninstall -y torch torchvision torchaudio
C:\ProgramData\anaconda3\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

说明：`torch` 的 CUDA 轮子体积较大（约 2.7GB+），下载需一些时间；中断后再次执行 `pip install` 通常会续传。

## 安装后验证

```powershell
C:\ProgramData\anaconda3\python.exe -c "import torch; print('version:', torch.__version__); print('cuda built:', torch.version.cuda); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

期望大致为：

- `version` 含 `+cu128`（或你选择的 CUDA 索引对应后缀）。
- `cuda built` 为 `12.8`（cu128 构建）。
- `cuda available` 为 `True`。

## 系统侧快速检查 GPU

```powershell
nvidia-smi
```

## 常见问题说明

1. **`Defaulting to user installation because normal site-packages is not writeable`**  
   pip 会把包装到用户目录，例如：  
   `C:\Users\<用户名>\AppData\Roaming\Python\Python312\site-packages`  
   这是正常现象。

2. **`torchrun.exe` 不在 PATH**  
   若要在命令行直接使用 `torchrun`，可将 pip 提示的 `...\Python312\Scripts` 加入用户 **PATH**。

3. **其他 CUDA 版本**  
   若需 **CUDA 12.6** 等，请把索引改为官网对应条目，例如：  
   `--index-url https://download.pytorch.org/whl/cu126`  
   具体以 [pytorch.org](https://pytorch.org/get-started/locally/) 上 **Windows + Pip + 你的 CUDA 版本** 为准。

4. **卸载残留目录**  
   若卸载时出现 `~orch` 等临时目录警告，可按提示手动删除该文件夹后再安装。

---

*文档中的 `C:\ProgramData\anaconda3\python.exe` 仅为示例，请按本机实际 Python 路径替换。*
