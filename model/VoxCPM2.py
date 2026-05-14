"""
VoxCPM2 TTS 服务封装（含 PyTorch 2.11+ CPU SDPA mask 补丁）。
"""

from __future__ import annotations

import gc
import inspect

_VOXCPM_SDPA_MASK_PATCHED: list[bool] = [False]


def _patch_voxcpm_minicpm_attention_sdpa_mask() -> None:
    """修复 PyTorch 2.11+ CPU：一维 attn_mask 触发 IndexError，展成 (1,1,1,seq) 后可正常广播。"""
    if _VOXCPM_SDPA_MASK_PATCHED[0]:
        return

    import torch
    from voxcpm.modules.minicpm4 import model as _minicpm_model

    def forward_step(self, hidden_states, position_emb, position_id, kv_cache):
        bsz, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, 1, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, 1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_emb is not None:
            cos, sin = position_emb
            query_states, key_states = _minicpm_model.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_cache, value_cache = kv_cache
        key_cache[:, :, position_id, :] = key_states
        value_cache[:, :, position_id, :] = value_states

        attn_mask_1d = torch.arange(key_cache.size(2), device=key_cache.device) <= position_id
        attn_mask = attn_mask_1d.view(1, 1, 1, -1)

        query_states = query_states.contiguous()
        key_cache = key_cache.contiguous()
        value_cache = value_cache.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_cache,
            value_cache,
            attn_mask=attn_mask,
            enable_gqa=True,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)
        return attn_output

    _minicpm_model.MiniCPMAttention.forward_step = forward_step
    _VOXCPM_SDPA_MASK_PATCHED[0] = True


class VoxCPM2Service:
    def __init__(self, model_path: str, *, load_denoiser: bool = False, device: str = "cpu"):
        import torch

        _patch_voxcpm_minicpm_attention_sdpa_mask()

        from voxcpm import VoxCPM

        self._model = VoxCPM.from_pretrained(model_path, load_denoiser=load_denoiser, local_files_only=True)
        if hasattr(self._model, "to"):
            try:
                self._model = self._model.to(device)
            except Exception as e:
                if device != "cpu":
                    print(f"[VoxCPM2] 模型切到 GPU 失败，回退 CPU：{type(e).__name__}: {e}", flush=True)
                    try:
                        self._model = self._model.to("cpu")
                    except Exception as e2:
                        print(f"[VoxCPM2] 模型切到 CPU 也失败：{type(e2).__name__}: {e2}", flush=True)
                else:
                    print(f"[VoxCPM2] 模型切到 CPU 失败：{type(e).__name__}: {e}", flush=True)

        self._sample_rate = int(self._model.tts_model.sample_rate)

    @property
    def backend(self) -> str:
        return "voxcpm"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def generate(
        self,
        text: str,
        reference_wav_path: str,
        *,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        seed: int = 0,
    ):
        """生成 TTS 音频，返回 numpy 数组 (N,) float32。"""
        self._apply_seed(seed)

        gen_kw: dict = dict(
            text=text,
            reference_wav_path=reference_wav_path,
            cfg_value=cfg_value,
            inference_timesteps=inference_timesteps,
        )
        if seed != 0:
            sig = inspect.signature(self._model.generate)
            if "seed" in sig.parameters:
                gen_kw["seed"] = int(seed)
            elif "random_seed" in sig.parameters:
                gen_kw["random_seed"] = int(seed)

        print(f"[VoxCPM2] ====== 提交参数 ======", flush=True)
        print(f"[VoxCPM2] backend: voxcpm", flush=True)
        print(f"[VoxCPM2] text: {text[:120]}{'…' if len(text) > 120 else ''}", flush=True)
        print(f"[VoxCPM2] reference_wav_path: {reference_wav_path}", flush=True)
        print(f"[VoxCPM2] cfg_value: {cfg_value}", flush=True)
        print(f"[VoxCPM2] inference_timesteps: {inference_timesteps}", flush=True)
        for k in ("seed", "random_seed"):
            if k in gen_kw:
                print(f"[VoxCPM2] {k}: {gen_kw[k]}", flush=True)
        if seed != 0 and "seed" not in gen_kw and "random_seed" not in gen_kw:
            print(f"[VoxCPM2] seed: {seed} (模型不支持，仅设全局种子)", flush=True)
        elif seed == 0:
            print(f"[VoxCPM2] seed: 0 (随机)", flush=True)
        print(f"[VoxCPM2] ========================", flush=True)

        wav = self._model.generate(**gen_kw)
        if isinstance(wav, (list, tuple)):
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
        np.random.seed(s % (2 ** 32))

    def release(self) -> None:
        import torch

        del self._model
        self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
