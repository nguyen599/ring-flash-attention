import torch
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward
from .ring_flash_attn import ring_flash_attn_backward
from einops import rearrange
from .utils import get_default_args, AllGatherComm as Comm
import logging
import torch.distributed._tensor as distp_tensor
import flash_attn
import os

if torch.__version__ >= "2.4.0" and flash_attn.__version__ >= "2.7.0":
    _wrapped_flash_attn_forward = torch.ops.flash_attn._flash_attn_forward
else:
    _wrapped_flash_attn_forward = _flash_attn_forward

if torch.__version__ >= "2.4.0":
    _wrapped_flash_attn_backward = torch.ops.flash_attn._flash_attn_backward
else:
    _wrapped_flash_attn_backward = _flash_attn_backward

def llama_flash_attn_forward(
    process_group,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads_k_stride,
    # local_k_slice,  # k slice is only meant for var_len (q_local only attend to k_slice)
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    time_event=None,  # Sync GPU,CPU to lower vRAM allocation; no sync by default
):
    out_list = []
    lse_list = []
    # logging.debug(f"bwd q {q[0,:2,0,:3]}")     

    nheads = q.shape[2]
    # total_k, nheads_k, head_dim = k.shape
    batch_k, seq_k, nheads_k, head_dim = k.shape
    assert nheads_k % heads_k_stride == 0

    world_size = dist.get_world_size(process_group)

    kv_buffer = torch.empty(
        (2, world_size, batch_k, seq_k, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )
    kv_buffer_copy = torch.empty_like(kv_buffer)

    k_0 = k[:, :, :heads_k_stride].contiguous()
    v_0 = v[:, :, :heads_k_stride].contiguous()

    comm = Comm(process_group)
    # Pass the main tensor slices to all_gather
    comm.all_gather(kv_buffer_copy[0], k_0)
    comm.all_gather(kv_buffer_copy[1], v_0)

    for i in range(0, nheads_k, heads_k_stride):
        # Optimization: No sync on last head stride
        if (i == nheads_k - heads_k_stride) and (time_event is not None):
            time_event.record()
        comm.wait()
        # Swap the main storage tensors
        kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer

        if i < nheads_k - heads_k_stride:
            # all_gather the next kv slice
            kv_slice_left = i + heads_k_stride
            kv_slice_right = kv_slice_left + heads_k_stride
            send_k = k[:,:,  kv_slice_left:kv_slice_right].contiguous()
            send_v = v[:,:,  kv_slice_left:kv_slice_right].contiguous()
            # Pass the main tensor slices for the next round
            comm.all_gather(kv_buffer_copy[0], send_k)
            comm.all_gather(kv_buffer_copy[1], send_v)

        q_i = q[:, :, i * nheads // nheads_k : (i + heads_k_stride) * nheads // nheads_k]
        # kv_buffer[0] has shape (batch_k, seq_k, world_size, heads_k_stride, head_dim)
        # We want k_i to be (batch_k, seq_k * world_size, heads_k_stride, head_dim)
        k_i = rearrange(kv_buffer[0], 'w b s hs dh -> b (w s) hs dh')
        v_i = rearrange(kv_buffer[1], 'w b s hs dh -> b (w s) hs dh')


        # params = get_default_args(_flash_attn_varlen_forward).copy()
        params = {
            "q": q_i,
            "k": k_i,
            "v": v_i,
            "dropout_p": dropout_p,
            "softmax_scale": softmax_scale,
            "causal": causal, # 'step' was not defined in this scope
            "window_size_left": window_size[0],
            "window_size_right": window_size[1],
            "softcap": softcap,
            "alibi_slopes": alibi_slopes,
            "return_softmax": True and dropout_p > 0,
        }
        # logging.debug(f"fwd i {i} k_ishape {k_i.shape} s{k_i[0,:3,0,:2]} e{k_i[0,-3:,0,:2]} q_i.shape {q_i.shape} params {params}")     
        # process_id = os.getpid()
        # if not os.path.exists('./logging/k_buffer_{}.pt'.format(process_id)):
        #     torch.save(k_i.detach(), './logging/k_buffer_{}.pt'.format(process_id))
        # out, _, _, _, _, lse, _, _ = _flash_attn_varlen_forward(**params)
        outputs = _wrapped_flash_attn_forward(**params)
        if len(outputs) == 8:
            out, _, _, _, _, lse, _, _ = outputs
        else:
            assert len(outputs) == 4
            out, lse, _, _ = outputs
        out_list.append(out)
        lse_list.append(lse)

    # out = torch.cat(out_list, dim=1)
    out = torch.cat(out_list, dim=2)
    # lse (B H S)
    lse = torch.cat(lse_list, dim=-2)
    return out, lse


def llama_flash_attn_backward(
    process_group,
    dout,
    q,
    k,
    v,
    out,
    softmax_lse,
    heads_k_stride,
    softmax_scale,
    dropout_p=0,
    causal=True,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
):
    nheads = q.shape[2]
    batch_k, seq_k, nheads_k, head_dim = k.shape
    assert nheads_k % heads_k_stride == 0

    world_size = dist.get_world_size(process_group)
    
    kv_buffer = torch.empty(
        (2, world_size, batch_k, seq_k, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    )
    kv_buffer_copy = torch.empty_like(kv_buffer)

    dkv_buffer = torch.empty(
        (2, batch_k, seq_k * world_size, heads_k_stride, head_dim),
        dtype=k.dtype,
        device=k.device,
    ) 

    if heads_k_stride != nheads_k:
        # for reduce_scatter_tensor
        kv_contiguous_buffer = torch.empty(
            (2, batch_k, seq_k, heads_k_stride, head_dim),
            dtype=k.dtype,
            device=k.device,
        )

    
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    comm = Comm(process_group)

    k_0 = k[:, :, :heads_k_stride].contiguous()
    v_0 = v[:, :, :heads_k_stride].contiguous()

    # Pass the main tensor slices to all_gather
    comm.all_gather(kv_buffer_copy[0], k_0)
    comm.all_gather(kv_buffer_copy[1], v_0)

    for i in range(0, nheads_k, heads_k_stride):
        dkv_buffer.zero_()

        q_slice = slice(
            i * nheads // nheads_k, (i + heads_k_stride) * nheads // nheads_k
        )
        q_i = q[:, :, q_slice]
        dout_i = dout[:, :, q_slice]
        out_i = out[:, :, q_slice]
        dq_i = dq[:, :, q_slice]
        if softmax_lse.dim() == 3:
            lse_i = softmax_lse[:, q_slice].contiguous()
        else:
            lse_i = softmax_lse[q_slice]

        comm.wait()
        # Swap the main storage tensors
        kv_buffer, kv_buffer_copy = kv_buffer_copy, kv_buffer
        # logging.debug(f"bwd i {i} q_slice {q_slice} kshape {k.shape} dv.shape {dv.shape} q.shape {q.shape}")     

        if i < nheads_k - heads_k_stride:
            # all_gather the next kv slice
            kv_slice_left = i + heads_k_stride
            kv_slice_right = kv_slice_left + heads_k_stride
            send_k = k[:, :, kv_slice_left:kv_slice_right].contiguous()
            send_v = v[:, :, kv_slice_left:kv_slice_right].contiguous()
            # Pass the main tensor slices for the next round
            comm.all_gather(kv_buffer_copy[0], send_k)
            comm.all_gather(kv_buffer_copy[1], send_v)

        # kv_buffer[0] has shape (batch_k, seq_k, world_size, heads_k_stride, head_dim)
        # We want k_i to be (batch_k, seq_k * world_size, heads_k_stride, head_dim)
        k_i = rearrange(kv_buffer[0], 'w b s hs dh -> b (w s) hs dh')
        v_i = rearrange(kv_buffer[1], 'w b s hs dh -> b (w s) hs dh')
        dk_i = dkv_buffer[0]
        dv_i = dkv_buffer[1]

        # params = get_default_args(_flash_attn_varlen_backward).copy()
        params = {
                "dout": dout_i,
                "q": q_i,
                "k": k_i,
                "v": v_i,
                "out": out_i,
                "softmax_lse": lse_i,
                "dq": dq_i,
                "dk": dk_i,
                "dv": dv_i,
                "dropout_p": dropout_p,
                "softmax_scale": softmax_scale,
                "causal": causal,
                "window_size_left": window_size[0],
                "window_size_right": window_size[1],
                "softcap": softcap,
                "alibi_slopes": alibi_slopes,
                "deterministic": deterministic,
        }
        # logging.debug(f"i {i} q {params['q'].shape} k {params['k'].shape} v {params['v'].shape} dout {params['dout'].shape} softmax_lse {params['softmax_lse'].shape}")     
        _wrapped_flash_attn_backward(**params)

        if heads_k_stride != nheads_k:
            # reduce_scatter needs contiguous buffer
            dk_i = kv_contiguous_buffer[0]
            dv_i = kv_contiguous_buffer[1]
        else:
            dk_i = dk
            dv_i = dv

        dist.reduce_scatter_tensor(dk_i, dkv_buffer[0], group=process_group)
        dist.reduce_scatter_tensor(dv_i, dkv_buffer[1], group=process_group)

        if heads_k_stride != nheads_k:
            dk[:, :, i : i + heads_k_stride] = dk_i
            dv[:, :, i : i + heads_k_stride] = dv_i

    return dq, dk, dv


class LlamaRingFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        heads_k_stride,
        # local_k_slice,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
        bwd_event_sync,
    ):
        time_event = torch.cuda.Event(enable_timing=False)
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = llama_flash_attn_forward(
            group,
            q,
            k,
            v,
            heads_k_stride=heads_k_stride,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
            time_event=time_event,
        )
        time_event.synchronize()
        # logging.debug(f"out {out[0,:2,3,:4]} out {softmax_lse[0,:2,:5]}")     
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        ctx.bwd_event_sync = bwd_event_sync
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        time_event = None
        if ctx.bwd_event_sync:
            time_event = torch.cuda.Event(enable_timing=False)
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = ring_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            # heads_k_stride=ctx.heads_k_stride,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
            time_event=time_event,
        )
        if ctx.bwd_event_sync:
            time_event.synchronize()
        # return dq, dk, dv, None, None, None, None, None, None, None, None
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None

class LlamaFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        heads_k_stride,
        # local_k_slice,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_softmax,
        group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = llama_flash_attn_forward(
            group,
            q,
            k,
            v,
            heads_k_stride=heads_k_stride,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # logging.debug(f"out {out[0,:2,3,:4]} softmax_lse {softmax_lse[0,:2,:5]}")     
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.heads_k_stride = heads_k_stride
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = llama_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            heads_k_stride=ctx.heads_k_stride,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        # return dq, dk, dv, None, None, None, None, None, None, None, None
        return dq, dk, dv, None, None, None, None, None, None, None, None, None


def llama_fwd_ring_bwd_flash_attn_func(
    q,
    k,
    v,
    heads_k_stride=1,  # default 1 always works, but need optimize
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    bwd_event_sync=False,
):
    return LlamaRingFlashAttnFunc.apply(
        q,
        k,
        v,
        heads_k_stride,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
        bwd_event_sync,
    )

def llama_flash_attn_func(
    q,
    k,
    v,
    heads_k_stride=1,  # default 1 always works, but need optimize
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    # logging.debug(f"q {q[0,:2,3,:4]}")
    return LlamaFlashAttnFunc.apply(
        q,
        k,
        v,
        heads_k_stride,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )

from .llama3_flash_attn_varlen import llama3_flash_attn_varlen_backward, llama3_flash_attn_varlen_forward, llama3_flash_attn_prepare_cu_seqlens, Llama3FlashAttnVarlenFunc


def llama3_flash_attn_varlen_custom_func(
    q,
    k,
    v,
    heads_k_stride,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
    mesh=None,
):
    '''
    The input is sharded in total_length dim
    for world size 4:
        rank 0 takes 1/2 of seq 1
        rank 1 takes 2/2 of seq 1
        rank 2 takes 1/2 of seq 2
        rank 3 takes 2/2 of seq 2
    That way each rank Q attends to corresonding k when one acutually use varlen
    inefficient all gather just a temp fix for testing
    '''
    batch_k, seq_k, nheads_k, head_dim = k.shape
    # logging.debug(f"q {q[0,:2,3,:4]}")
    q_k_v = []
    for t in (q, k, v):
        q_k_v.append(
            distp_tensor.DTensor.from_local(
                t, mesh, [distp_tensor.Shard(1)]
            ).redistribute(mesh, [distp_tensor.Replicate()]).view(-1, nheads_k, head_dim
            ).redistribute(mesh, [distp_tensor.Shard(0)]).to_local() 
        )
    q, k, v = q_k_v
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)
    cu_seqlens = torch.arange(0, (batch_k + 1) * seq_k*world_size, step=seq_k*world_size,
                              dtype=torch.int32, device=k.device)
    (cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, local_k_slice
    ) = llama3_flash_attn_prepare_cu_seqlens(cu_seqlens, causal, rank, world_size)
    # logging.debug(f"{cu_seqlens_q}, {cu_seqlens_k}, {max_seqlen_q}, {max_seqlen_k}, {local_k_slice}")
    # logging.debug(f"{cu_seqlens}, {causal}, {rank}, {world_size}")
    # q, k, v = [rearrange(t, 'b s h d -> (b s) h d', b=) for t in (q, k, v)]
    # k = k.contiguous().view(-1,  nheads_k, head_dim)
    # v = v.contiguous().view(-1,  nheads_k, head_dim)
    # q = q.contiguous().view(-1,  nheads_k, head_dim)
    output = Llama3FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        heads_k_stride,
        local_k_slice,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )
    if return_attn_probs:
        (out, softmax_lse, none) = output
        out = distp_tensor.DTensor.from_local(
                out, mesh, [distp_tensor.Shard(0)]
            ).redistribute(mesh, [distp_tensor.Replicate()]).view(batch_k, seq_k*world_size, nheads_k, head_dim
            ).redistribute(mesh, [distp_tensor.Shard(1)]).to_local() 
        return out, softmax_lse, none
    else:
        output = distp_tensor.DTensor.from_local(
                output, mesh, [distp_tensor.Shard(0)]
            ).redistribute(mesh, [distp_tensor.Replicate()]).view(batch_k, seq_k*world_size, nheads_k, head_dim
            ).redistribute(mesh, [distp_tensor.Shard(1)]).to_local() 
        logging.debug(f" out {output[0,:2,3,:4]}")     
        return output
