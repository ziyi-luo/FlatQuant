import triton
import triton.language as tl
import torch
import math
import numpy as np
from triton.language.extra import libdevice
import deploy
from deploy.nn.quantization import Quantizer


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),  
        triton.Config({'BLOCK_SIZE_K': 128}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE_K': 128}, num_stages=1, num_warps=4),
    ],
    key=['B', 'M', 'N'],
)


@triton.jit
def matmul_quant_kernel(
        b_ptr, c_ptr,
        res_ptr,
        output_scale,
        B,
        M: tl.constexpr, 
        N: tl.constexpr,
        np2_M: tl.constexpr, 
        np2_N: tl.constexpr,
        stride_bb, stride_bk, stride_bn,  
        stride_ck, stride_cn,
        stride_resb, stride_resm, stride_resn,
        BLOCK_SIZE_K: tl.constexpr,
):
    """
    Quant(b @ c)

    b [B, M, N]
    c [N, N]
    """

    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1) + tl.program_id(axis=2) * tl.num_programs(axis=1)
    pid_m = pid

    offs_bm = (pid_m * M + tl.arange(0, np2_M)) % M
    offs_cn = (tl.arange(0, np2_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    b_ptrs = b_ptr + batch_id * stride_bb.to(tl.int64) + (offs_bm[:, None] * stride_bk + offs_k[None, :] * stride_bn)
    c_ptrs = c_ptr + (offs_k[:, None] * stride_ck + offs_cn[None, :] * stride_cn)

    accumulator = tl.zeros((np2_M, np2_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BLOCK_SIZE_K)):
        b = tl.load(b_ptrs, mask=offs_k[None, :] < N - k * BLOCK_SIZE_K, other=0.0)
        c = tl.load(c_ptrs, mask=offs_k[:, None] < N - k * BLOCK_SIZE_K, other=0.0)    
        accumulator = tl.dot(b, c, accumulator)
        b_ptrs += BLOCK_SIZE_K * stride_bn
        c_ptrs += BLOCK_SIZE_K * stride_ck

    abs_src_val = tl.abs(accumulator)
    max_src_val = tl.max(abs_src_val)

    scale = max_src_val / 7.
    quant_val = libdevice.llrint(accumulator / scale)
    quant_val = max(-8, min(quant_val, 7))

    quant_val = quant_val.reshape(np2_M, np2_N // 2, 2, can_reorder=False)
    quant_val_even, quant_val_odd = quant_val.split()
    quant_val_odd = quant_val_odd << 4

    res = tl.zeros((np2_M, np2_N // 2), dtype=tl.int8)
    res = res | (quant_val_odd & 0xf0)
    res = res | (quant_val_even & 0x0f)

    offs_resm = pid_m * M + tl.arange(0, np2_M)
    offs_resn = tl.arange(0, np2_N // 2)
    res_ptrs = res_ptr + stride_resb.to(tl.int64) * batch_id + stride_resm * offs_resm[:, None] + stride_resn * offs_resn[None, :]
    res_mask = (offs_resm[:, None] < M) & (offs_resn[None, :] < N // 2)
    tl.store(res_ptrs, res, mask=res_mask)
    tl.store(output_scale + batch_id, scale.to(tl.float16))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_SIZE_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),  
        triton.Config({'BLOCK_SIZE_K': 128}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE_K': 128}, num_stages=1, num_warps=4),
    ],
    key=['B', 'M', 'N'],
)


@triton.jit
def matmul_kernel(
        b_ptr, c_ptr,
        res_ptr,
        output_scale,
        B,
        M: tl.constexpr, 
        N: tl.constexpr,
        np2_M: tl.constexpr, 
        np2_N: tl.constexpr,
        stride_bb, stride_bk, stride_bn,  
        stride_ck, stride_cn,
        stride_resb, stride_resm, stride_resn,
        BLOCK_SIZE_K: tl.constexpr,
):
    """
    Quant(b @ c)

    b [B, M, N]
    c [N, N]
    """

    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1) + tl.program_id(axis=2) * tl.num_programs(axis=1)
    pid_m = pid

    offs_bm = (pid_m * M + tl.arange(0, np2_M)) % M
    offs_cn = (tl.arange(0, np2_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    b_ptrs = b_ptr + batch_id * stride_bb.to(tl.int64) + (offs_bm[:, None] * stride_bk + offs_k[None, :] * stride_bn)
    c_ptrs = c_ptr + (offs_k[:, None] * stride_ck + offs_cn[None, :] * stride_cn)

    accumulator = tl.zeros((np2_M, np2_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(N, BLOCK_SIZE_K)):
        b = tl.load(b_ptrs, mask=offs_k[None, :] < N - k * BLOCK_SIZE_K, other=0.0)
        c = tl.load(c_ptrs, mask=offs_k[:, None] < N - k * BLOCK_SIZE_K, other=0.0)    
        accumulator = tl.dot(b, c, accumulator)
        b_ptrs += BLOCK_SIZE_K * stride_bn
        c_ptrs += BLOCK_SIZE_K * stride_ck

    offs_resm = pid_m * M + tl.arange(0, np2_M)
    offs_resn = tl.arange(0, np2_N)
    res_ptrs = res_ptr + stride_resb.to(tl.int64) * batch_id + stride_resm * offs_resm[:, None] + stride_resn * offs_resn[None, :]
    res_mask = (offs_resm[:, None] < M) & (offs_resn[None, :] < N // 2)
    tl.store(res_ptrs, accumulator.to(tl.float16), mask=res_mask)


@triton.jit
def quant_kernel(
        src_ptr,
        stride_srcb, stride_srcm, stride_srcn,
        dst_ptr,
        stride_dstb, stride_dstm, stride_dstn,
        output_scale,
        B,
        M: tl.constexpr, 
        N: tl.constexpr,
        np2_M: tl.constexpr, 
        np2_N: tl.constexpr,
):
    '''
    quant fp16 tensor to int4
    '''
    batch_id = tl.program_id(axis=0) + tl.program_id(axis=1) * tl.num_programs(axis=0)
    index_rows = tl.arange(0, np2_M)
    index_cols = tl.arange(0, np2_N)

    src_ptrs = src_ptr + batch_id * stride_srcb.to(tl.int64) + index_rows[:, None] * stride_srcm + index_cols[None, :] * stride_srcn
    src_mask = (index_rows[:, None] < M) & (index_cols[None, :] < N)
    src = tl.load(src_ptrs, mask=src_mask, other=0.0)

    abs_src_val = tl.abs(src)
    max_src_val = tl.max(abs_src_val)
    scale = max_src_val / 7.

    quant_val = libdevice.llrint(src / scale)
    quant_val = max(-8, min(quant_val, 7))
    quant_val = quant_val.reshape(np2_M,  np2_N // 2, 2, can_reorder=False)
    quant_val_even, quant_val_odd = quant_val.split()
    quant_val_odd = quant_val_odd << 4

    res = tl.zeros((np2_M, np2_N // 2), dtype=tl.uint8)
    res = res | (quant_val_odd & 0xf0)
    res = res | (quant_val_even & 0x0f)

    offs_resm = tl.arange(0, np2_M)
    offs_resn = tl.arange(0, np2_N // 2)
    dst_ptrs = dst_ptr + stride_dstb.to(tl.int64) * batch_id + stride_dstm * offs_resm[:, None] + stride_dstn * offs_resn[None, :]
    res_mask = (offs_resm[:, None] < M) & (offs_resn[None, :] < N // 2)
    tl.store(dst_ptrs, res, mask=res_mask)
    tl.store(output_scale + batch_id, scale)


FUSION=True
def block_matmul(b, c, seq_len):
    # Check constraints.
    # b @ c, b [b, m, n], c [n, n]
    assert b.shape[2] == c.shape[0], "Incompatible dimensions"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert c.is_contiguous(), "Matrix C must be contiguous"
    B, M, N = b.shape
    Actual_B = B // seq_len
    BLOCK_SIZE_M = triton.next_power_of_2(M)
    # Allocates output.
    output_scale = torch.empty((B, 1), device=b.device, dtype=torch.float16)
    quant_res = torch.empty((B, M, N // 2), device=b.device, dtype=torch.uint8)

    # 1D launch kernel where each block gets its own program.
    if FUSION:
        grid = (1, seq_len, Actual_B)
        matmul_quant_kernel[grid](
            b, c,  #
            quant_res, #
            output_scale, #
            B, M, N,  #
            triton.next_power_of_2(M),
            triton.next_power_of_2(N),
            b.stride(0), b.stride(1), b.stride(2),  #
            c.stride(0), c.stride(1), #
            quant_res.stride(0), quant_res.stride(1), quant_res.stride(2),  #
        )
    else:
        bmm_res = torch.empty((B, M, N), device=b.device, dtype=b.dtype)
        grid = (1, seq_len, Actual_B)
        matmul_kernel[grid](
            b, c,  #
            bmm_res, #
            output_scale, #
            B, M, N,  #
            triton.next_power_of_2(M),
            triton.next_power_of_2(N),
            b.stride(0), b.stride(1), b.stride(2),  #
            c.stride(0), c.stride(1), #
            bmm_res.stride(0), bmm_res.stride(1), bmm_res.stride(2),  #
        )
        grid = (seq_len, Actual_B)
        quant_kernel[grid](
            bmm_res,
            bmm_res.stride(0), bmm_res.stride(1), bmm_res.stride(2), 
            quant_res,
            quant_res.stride(0), quant_res.stride(1), quant_res.stride(2),
            output_scale,
            B, M, N,
            triton.next_power_of_2(M),
            triton.next_power_of_2(N),
        )
    packed_tensor = deploy.PackedQuantizedTensor(quant_res.reshape(B, -1), output_scale)
    return packed_tensor


def benchmark(B, M, N, S, provider):
    # B = Batch * SeqLen
    # Use device from utils for NPU support
    from flatquant.utils import DEV
    b = torch.randn((B, M, N), device=DEV, dtype=torch.float16)
    c = torch.randn((N, N), device=DEV, dtype=torch.float16)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'cublas':
        quantizer = Quantizer()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: quantizer(torch.matmul(b, c).view(B, -1)), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: block_matmul(b, c, S), quantiles=quantiles)
    perf = lambda ms: 2 * B * M * N * N * 1e-12 / (ms * 1e-3)
    import pdb; pdb.set_trace()
    return perf(ms), perf(max_ms), perf(min_ms), ms, max_ms, min_ms
