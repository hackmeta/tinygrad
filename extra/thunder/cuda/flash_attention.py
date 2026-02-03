"""CUDA Flash Attention for NVIDIA GPUs using ThunderKittens

Supported architectures:
- Ampere (sm_80, sm_86, sm_87): A100, RTX 30xx
- Ada (sm_89): RTX 40xx
- Hopper (sm_90): H100
"""
import re
import pathlib
from tinygrad import Device, Tensor, dtypes
from tinygrad.runtime.support.compiler_cuda import NVCCCompiler

import threading

# LRU cache to prevent unbounded growth (max 32 different kernel configs)
_KERNEL_CACHE_SIZE = 32
_kernel_cache = {}
_kernel_cache_lock = threading.Lock()

def _get_kittens_flag(arch: str) -> str:
    """Select ThunderKittens flag based on GPU architecture."""
    # Validate arch format (e.g., "sm_86", "sm_89")
    match = re.match(r"sm_(\d+)", arch)
    if not match:
        raise RuntimeError(f"Invalid GPU architecture format: {arch!r}, expected sm_XX")
    sm_version = int(match.group(1))

    if sm_version >= 100:
        return "-DKITTENS_BLACKWELL"
    elif sm_version >= 90:
        return "-DKITTENS_HOPPER"
    elif sm_version >= 89:
        return "-DKITTENS_4090"
    elif sm_version >= 80:
        return "-DKITTENS_A100"
    else:
        raise RuntimeError(f"Flash attention requires Ampere+ GPU (sm_80+), got {arch}")

def _get_kernel(B: int, N: int, H: int, D: int = 64, is_causal: bool = False):
    device = Device["CUDA"]
    arch = device.compiler.arch

    key = (B, N, H, D, is_causal, arch)

    # Thread-safe cache access
    with _kernel_cache_lock:
        if key in _kernel_cache:
            return _kernel_cache[key]

    kittens_flag = _get_kittens_flag(arch)
    code = (pathlib.Path(__file__).parent / "fa_flex.cu").read_text()
    include_path = (pathlib.Path(__file__).parent / "include").as_posix()

    kitten_args = [
        f"-I{include_path}",
        "-std=c++20", "--expt-relaxed-constexpr", "--extended-lambda",
        kittens_flag,
        f"-DFA_B={B}", f"-DFA_N={N}", f"-DFA_H={H}", f"-DFA_D={D}"
    ]
    lib = NVCCCompiler(arch, kitten_args).compile(code)

    # Find kernel by exact name matching (more robust than substring)
    # Non-causal kernel: "attend_ker" but NOT containing "causal"
    # Causal kernel: "attend_ker_causal"
    kernel_name = None
    found_globals = []
    ptx_text = lib.decode()

    for line in ptx_text.split("\n"):
        if ".globl" in line:
            name = line.split()[-1]
            found_globals.append(name)

    # Search for exact match first, then partial match
    if is_causal:
        # Look for attend_ker_causal (exact or mangled)
        for name in found_globals:
            if "attend_ker_causal" in name:
                kernel_name = name
                break
    else:
        # Look for attend_ker but NOT causal version
        for name in found_globals:
            if "attend_ker" in name and "causal" not in name:
                kernel_name = name
                break

    if kernel_name is None:
        diag = f"is_causal={is_causal}, Found globals: {found_globals[:10]}"
        if len(found_globals) > 10:
            diag += f" (and {len(found_globals)-10} more)"
        raise RuntimeError(f"Could not find {'causal' if is_causal else 'non-causal'} kernel. {diag}")

    prg = device.runtime(kernel_name, lib)
    # Dynamic shared memory based on architecture and kernel requirements
    # k_smem + v_smem: LOAD_BLOCKS(2) * PIPE_STAGES(3) * tile_size each
    # tile_size for D=64: ROWS<64>=16, so 16*64*2 bytes = 2KB per tile
    # Total: 2 * 2 * 3 * 2KB = 24KB, plus some overhead -> 48KB is safe
    smem_size = 16384 * 3  # 48KB - conservative default for all architectures
    prg.smem = smem_size

    # Thread-safe cache store with size limit
    with _kernel_cache_lock:
        if len(_kernel_cache) >= _KERNEL_CACHE_SIZE:
            oldest_key = next(iter(_kernel_cache))
            del _kernel_cache[oldest_key]
        _kernel_cache[key] = prg
    return prg

def flash_attention(q: Tensor, k: Tensor, v: Tensor, attn_mask=None, is_causal: bool = False) -> Tensor:
    """Flash attention for NVIDIA GPUs (Ampere+).

    Input shape: [B, N, H, D] (batch, sequence, heads, head_dim)
    Output shape: [B, N, H, D] (contiguous)

    Requirements:
    - Input must be 4D with shape [B, N, H, D]
    - D must be 64
    - N must be >= 64 and divisible by 64
    - q, k, v must have identical shapes
    - GPU must be Ampere or newer (sm_80+)
    """
    if attn_mask is not None:
        raise NotImplementedError("attn_mask not supported in CUDA flash attention, caller should handle fallback")

    # Strict shape validation
    if q.ndim != 4:
        raise ValueError(f"Expected 4D tensor [B,N,H,D], got {q.ndim}D with shape {q.shape}")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"q, k, v must have identical shapes. Got q={q.shape}, k={k.shape}, v={v.shape}")

    # Device consistency check
    if not (q.device == k.device == v.device):
        raise ValueError(f"q, k, v must be on same device. Got q={q.device}, k={k.device}, v={v.device}")
    if q.device is None or "CUDA" not in str(q.device):
        raise ValueError(f"Tensors must be on CUDA device, got {q.device}")

    B, N, H, D = q.shape

    if D != 64:
        raise ValueError(f"D must be 64, got {D}")
    if N < 64 or N % 64 != 0:
        raise ValueError(f"N must be >= 64 and divisible by 64, got {N}")

    # Cast to bf16 if needed
    if q.dtype != dtypes.bfloat16: q = q.cast(dtypes.bfloat16)
    if k.dtype != dtypes.bfloat16: k = k.cast(dtypes.bfloat16)
    if v.dtype != dtypes.bfloat16: v = v.cast(dtypes.bfloat16)

    out = Tensor.empty(B, N, H, D, device="CUDA", dtype=dtypes.bfloat16)
    Tensor.realize(q, k, v, out)

    prg = _get_kernel(B, N, H, D, is_causal)
    gsz = (N // 64, H, B)
    lsz = (128, 1, 1)

    prg(out.uop.buffer.ensure_allocated()._buf,
        q.uop.buffer._buf, k.uop.buffer._buf, v.uop.buffer._buf,
        global_size=gsz, local_size=lsz, wait=True)

    return out
