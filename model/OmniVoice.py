"""
OmniVoice TTS 服务封装。
映射 VoxCPM2 参数（cfg_value / inference_timesteps / seed）到 OmniVoiceGenerationConfig。
"""

from __future__ import annotations

import gc
import re


def _detect_language(text: str) -> str:
    """根据文本内容检测语言，避免语言无关模式下的随机跳动。"""
    if re.search(r"[一-鿿]", text):
        return "zh"
    if re.search(r"[぀-ゟ゠-ヿ]", text):
        return "ja"
    if re.search(r"[가-힯]", text):
        return "ko"
    return "en"


class OmniVoiceService:
    def __init__(self, model_path: str, device_map: str = "cpu", dtype=None):
        import torch

        from omnivoice import OmniVoice, OmniVoiceGenerationConfig

        if dtype is None:
            dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self._model = OmniVoice.from_pretrained(model_path, device_map=device_map, dtype=dtype, local_files_only=True)
        self._sample_rate = 24000
        self._OmniVoiceGenerationConfig = OmniVoiceGenerationConfig

    @property
    def backend(self) -> str:
        return "omnivoice"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def generate(
        self,
        text: str,
        reference_wav_path: str,
        *,
        cfg_value: float = 2.0,
        inference_timesteps: int = 32,
        seed: int = 0,
    ):
        """生成 TTS 音频，返回 numpy 数组 (N,) float32。

        参数映射：
          cfg_value          → OmniVoiceGenerationConfig.guidance_scale
          inference_timesteps → OmniVoiceGenerationConfig.num_step
          seed              → 在推理前设置全局随机种子（0 = 不设）
        """
        self._apply_seed(seed)

        language = _detect_language(text)

        # OmniVoiceGenerationConfig 各参数说明：
        #
        # num_step (32) — 迭代解码步数。
        #   每一步逐个揭掩码（mask）预测音频 token。步数越多生成越精细，
        #   但线性增加耗时。← 映射自前端 inference_timesteps（默认 32）。
        #
        # guidance_scale (2.0) — 无分类器引导（CFG）强度。
        #   值越大模型越严格跟随文本，但过高会产生失真/杂音。
        #   0 = 关闭 CFG。← 映射自前端 cfg_value（默认 2.0）。
        #
        # t_shift (0.1) — 时间步偏移。
        #   控制掩码揭开的节奏：越小，早期（高噪声）步占比越大，
        #   生成更偏「探索」；越大则后期（低噪声）步越多，偏「精修」。
        #
        # layer_penalty_factor (5.0) — 音频编码层惩罚系数。
        #   OmniVoice 用 8 层 codebook 编码音频；值越大，底层（低频）
        #   优先揭掩码，高层（高频细节）延后，影响音质细腻度。
        #
        # position_temperature (5.0) — 位置选择温度。
        #   控制每步选择「哪个位置」揭掩码的随机程度。越高越随机，
        #   越低越倾向于置信度最高的位置（可能过于保守）。
        #
        # class_temperature (0.0) — token 采样温度。
        #   0 = 贪心（greedy），始终取概率最大的 token，结果可复现。
        #   >0 = 带随机性的采样，值越大输出多样性越高。
        #
        # denoise (True) — 是否前置 <|denoise|> token。
        #   True 时告诉模型「参考音频可能含噪」，让模型学会忽略背景噪声。
        #   语音克隆场景建议保持 True。
        #
        # preprocess_prompt (True) — 是否预处理参考音频。
        #   True 时对参考音频做静音切除 + 过长裁剪（>20s 按最大静音间隙拆分）。
        #   如果参考音频已经精修过可设为 False 跳过。
        #
        # postprocess_output (True) — 是否后处理生成音频。
        #   True 时自动去除长静音、淡入淡出、边缘补零、参考 RMS 音量对齐。
        #
        # audio_chunk_duration (15.0) — 分块时长（秒）。
        #   对于长文本，按此目标时长拆成多个 chunk 分别生成再拼接。
        #   设 0 或负数可禁用分块（长文本可能 OOM）。
        #
        # audio_chunk_threshold (30.0) — 分块触发阈值（秒）。
        #   仅当估算音频时长超过此值时才会分块。小于此值的文本一次性生成。
        ref_text = ""
        gen_config = self._OmniVoiceGenerationConfig(
            num_step=int(inference_timesteps),
            guidance_scale=float(cfg_value),
        )

        print(f"[OmniVoice] ====== 提交参数 ======", flush=True)
        print(f"[OmniVoice] backend: omnivoice", flush=True)
        print(f"[OmniVoice] text: {text[:120]}{'…' if len(text) > 120 else ''}", flush=True)
        print(f"[OmniVoice] language: {language}  (auto-detected)", flush=True)
        print(f"[OmniVoice] ref_audio: {reference_wav_path}", flush=True)
        print(f"[OmniVoice] ref_text: {ref_text}", flush=True)
        print(f"[OmniVoice] seed: {seed}", flush=True)
        print(f"[OmniVoice] --- OmniVoiceGenerationConfig ---", flush=True)
        print(f"[OmniVoice]   num_step              = {gen_config.num_step}  ← inference_timesteps", flush=True)
        print(f"[OmniVoice]   guidance_scale        = {gen_config.guidance_scale}  ← cfg_value", flush=True)
        print(f"[OmniVoice]   t_shift               = {gen_config.t_shift}", flush=True)
        print(f"[OmniVoice]   layer_penalty_factor  = {gen_config.layer_penalty_factor}", flush=True)
        print(f"[OmniVoice]   position_temperature  = {gen_config.position_temperature}", flush=True)
        print(f"[OmniVoice]   class_temperature     = {gen_config.class_temperature}", flush=True)
        print(f"[OmniVoice]   denoise               = {gen_config.denoise}", flush=True)
        print(f"[OmniVoice]   preprocess_prompt     = {gen_config.preprocess_prompt}", flush=True)
        print(f"[OmniVoice]   postprocess_output    = {gen_config.postprocess_output}", flush=True)
        print(f"[OmniVoice]   audio_chunk_duration  = {gen_config.audio_chunk_duration}", flush=True)
        print(f"[OmniVoice]   audio_chunk_threshold = {gen_config.audio_chunk_threshold}", flush=True)
        print(f"[OmniVoice] ========================", flush=True)

        # ref_text="" 阻止自动加载 Whisper ASR 转录，且空字符串不会被拼接到合成文本
        wav = self._model.generate(
            text=text,
            language=language,
            ref_text="",
            ref_audio=reference_wav_path,
            generation_config=gen_config,
        )
        if isinstance(wav, list):
            wav = wav[0]
        return wav

    @staticmethod
    def _apply_seed(seed: int) -> None:
        if seed == 0:
            return
        import random

        import numpy as np
        import torch

        s = int(seed)
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        random.seed(s & ((1 << 63) - 1))
        np.random.seed(s % (2**32))

    def release(self) -> None:
        import torch

        del self._model
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
