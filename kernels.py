import torch
import triton
import triton.language as tl


# ============================================================
# A1: DSD matmul  block-sparse A @ dense B -> dense C
# ============================================================

@triton.jit
def _dsd_kernel(
    values_ptr, row_offsets_ptr, col_indices_ptr,
    B_ptr, C_ptr,
    M, K, N, block: tl.constexpr,
    sv0, sv1, sv2,   # strides for values
    sb0, sb1,        # strides for B
    sc0, sc1,        # strides for C
    BLOCK_N: tl.constexpr,
):
    pid_row = tl.program_id(0)  # 第几个block-row
    pid_col = tl.program_id(1)  # 沿着N方向的第几个tile

    row_start = pid_row * block
    col_start = pid_col * BLOCK_N

    offs_m = row_start + tl.arange(0, block)    # 这个block-row对应的output行
    offs_n = col_start + tl.arange(0, BLOCK_N)  # 这个tile对应的output列

    acc = tl.zeros((block, BLOCK_N), dtype=tl.float32)  # accumulator

    lo = tl.load(row_offsets_ptr + pid_row)      # 这一行的live block从哪开始
    hi = tl.load(row_offsets_ptr + pid_row + 1)  # 到哪结束

    for idx in range(lo, hi):
        k_block = tl.load(col_indices_ptr + idx)  # 这个live block在K方向的位置
        k_start = k_block * block

        # 把A的这个block load进来，shape是(block, block)
        a_ptrs = (values_ptr
                  + idx * sv0
                  + tl.arange(0, block)[:, None] * sv1
                  + tl.arange(0, block)[None, :] * sv2)
        a_blk = tl.load(a_ptrs)

        # load对应的B tile，shape是(block, BLOCK_N)
        offs_k = k_start + tl.arange(0, block)
        b_ptrs = B_ptr + offs_k[:, None] * sb0 + offs_n[None, :] * sb1
        b_blk = tl.load(b_ptrs,
                         mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
                         other=0.0)

        acc += tl.dot(a_blk, b_blk, allow_tf32=False)  # 题目要求不能用tf32

    c_ptrs = C_ptr + offs_m[:, None] * sc0 + offs_n[None, :] * sc1
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def dsd_matmul(values, row_offsets, column_indices, B, M, K, N, block):
    C = torch.zeros(M, N, device=B.device, dtype=torch.float32)

    BLOCK_N = min(128, triton.next_power_of_2(N))
    grid = (M // block, triton.cdiv(N, BLOCK_N))

    _dsd_kernel[grid](
        values, row_offsets, column_indices, B, C,
        M, K, N, block,
        values.stride(0), values.stride(1), values.stride(2),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_N=BLOCK_N,
    )
    return C


# ============================================================
# A2: sparse flash attention forward
# ============================================================

@triton.jit
def _fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    q_row_offsets_ptr, q_col_indices_ptr,
    sm_scale,
    sqb, sqt, sqd,   # Q的strides：per head, per token, per dim
    skb, skt, skd,
    svb, svt, svd,
    sob, sot, sod,
    slb, slt,
    T, d: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    LOG2E: tl.constexpr,
):
    pid_bh = tl.program_id(0)  # 第几个(batch, head)，已经flatten了
    pid_q  = tl.program_id(1)  # 第几个query block

    q_start = pid_q * BLOCK_Q
    offs_q = q_start + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, d)
    mask_q = offs_q[:, None] < T

    # 把这个head的Q block load进来
    q = tl.load(Q_ptr + pid_bh * sqb + offs_q[:, None] * sqt + offs_d[None, :] * sqd,
                mask=mask_q, other=0.0).to(tl.float32)

    # online softmax需要的状态，初始化
    m_i = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)  # 当前最大值
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)                # 分母累加
    acc = tl.zeros((BLOCK_Q, d), dtype=tl.float32)              # output累加

    lo = tl.load(q_row_offsets_ptr + pid_q)
    hi = tl.load(q_row_offsets_ptr + pid_q + 1)

    for idx in range(lo, hi):
        k_block = tl.load(q_col_indices_ptr + idx)  # 这个query block要attend的key block
        k_start = k_block * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k[:, None] < T

        k = tl.load(K_ptr + pid_bh * skb + offs_k[:, None] * skt + offs_d[None, :] * skd,
                    mask=mask_k, other=0.0).to(tl.float32)
        v = tl.load(V_ptr + pid_bh * svb + offs_k[:, None] * svt + offs_d[None, :] * svd,
                    mask=mask_k, other=0.0).to(tl.float32)

        scores = tl.dot(q, tl.trans(k), allow_tf32=True) * sm_scale  # (BLOCK_Q, BLOCK_K)
        scores = tl.where(offs_k[None, :] < T, scores, float('-inf'))  # 超出范围的key干掉

        # online softmax update，用base-2算，数值上更稳
        s2 = scores * LOG2E
        m_new = tl.maximum(m_i, tl.max(s2, axis=1))
        alpha = tl.exp2(m_i - m_new)       # 旧的acc要scale down
        p = tl.exp2(s2 - m_new[:, None])   # 当前block的attention weights

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16).to(tl.float32), v, allow_tf32=True)
        m_i = m_new

    o = acc / l_i[:, None]  # normalize

    # L存的是log2(分母)，反向传播会用到
    L_val = tl.log2(l_i) + m_i

    tl.store(O_ptr + pid_bh * sob + offs_q[:, None] * sot + offs_d[None, :] * sod,
             o.to(tl.float16), mask=mask_q)
    tl.store(L_ptr + pid_bh * slb + offs_q * slt,
             L_val, mask=offs_q < T)


def sparse_flash_forward(Q, K, V, q_row_offsets, q_col_indices,
                         sm_scale, BLOCK_Q, BLOCK_K):
    B, H, T, d = Q.shape
    O = torch.empty_like(Q)
    L = torch.empty(B, H, T, device=Q.device, dtype=torch.float32)

    LOG2E = 1.4426950408889634
    grid = (B * H, T // BLOCK_Q)

    _fwd_kernel[grid](
        Q, K, V, O, L,
        q_row_offsets, q_col_indices,
        sm_scale,
        Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(1), K.stride(2), K.stride(3),
        V.stride(1), V.stride(2), V.stride(3),
        O.stride(1), O.stride(2), O.stride(3),
        L.stride(1), L.stride(2),
        T, d,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K,
        LOG2E=LOG2E,
        num_warps=4,
    )
    return O, L


# ============================================================
# A3: sparse flash attention backward
# ============================================================

@triton.jit
def _bwd_preprocess(
    O_ptr, dO_ptr, D_ptr,
    sob, sot, sod,
    sdob, sdot, sdod,
    sdb, sdt,
    T, d: tl.constexpr,
    BLOCK_Q: tl.constexpr,
):
    # 先算D_i = rowsum(dO * O)，后面算dS的时候要用
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, d)
    mask = offs_q[:, None] < T

    o  = tl.load(O_ptr  + pid_bh * sob  + offs_q[:, None] * sot  + offs_d[None, :] * sod,
                 mask=mask, other=0.0).to(tl.float32)
    do = tl.load(dO_ptr + pid_bh * sdob + offs_q[:, None] * sdot + offs_d[None, :] * sdod,
                 mask=mask, other=0.0).to(tl.float32)

    D = tl.sum(o * do, axis=1)  # element-wise乘然后按行求和
    tl.store(D_ptr + pid_bh * sdb + offs_q * sdt, D, mask=offs_q < T)


@triton.jit
def _bwd_dq_kernel(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, D_ptr, dO_ptr, dQ_ptr,
    q_row_offsets_ptr, q_col_indices_ptr,
    sm_scale,
    sqb, sqt, sqd,
    skb, skt, skd,
    svb, svt, svd,
    slb, slt,
    sdb, sdt,
    sdob, sdot, sdod,
    sdqb, sdqt, sdqd,
    T, d: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
    LOG2E: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, d)
    mask_q = offs_q[:, None] < T

    q  = tl.load(Q_ptr  + pid_bh * sqb  + offs_q[:, None] * sqt  + offs_d[None, :] * sqd,
                 mask=mask_q, other=0.0).to(tl.float32)
    do = tl.load(dO_ptr + pid_bh * sdob + offs_q[:, None] * sdot + offs_d[None, :] * sdod,
                 mask=mask_q, other=0.0).to(tl.float32)
    Li = tl.load(L_ptr  + pid_bh * slb  + offs_q * slt, mask=offs_q < T, other=0.0)
    Di = tl.load(D_ptr  + pid_bh * sdb  + offs_q * sdt, mask=offs_q < T, other=0.0)

    dq_acc = tl.zeros((BLOCK_Q, d), dtype=tl.float32)

    lo = tl.load(q_row_offsets_ptr + pid_q)
    hi = tl.load(q_row_offsets_ptr + pid_q + 1)

    for idx in range(lo, hi):
        k_block = tl.load(q_col_indices_ptr + idx)
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k[:, None] < T

        k = tl.load(K_ptr + pid_bh * skb + offs_k[:, None] * skt + offs_d[None, :] * skd,
                    mask=mask_k, other=0.0).to(tl.float32)
        v = tl.load(V_ptr + pid_bh * svb + offs_k[:, None] * svt + offs_d[None, :] * svd,
                    mask=mask_k, other=0.0).to(tl.float32)

        # 重新算P，因为forward没把attention matrix存下来
        scores = tl.dot(q, tl.trans(k), allow_tf32=True) * sm_scale
        scores = tl.where(offs_k[None, :] < T, scores, float('-inf'))
        p = tl.exp2(scores * LOG2E - Li[:, None])  # 恢复出attention weights

        dp = tl.dot(do, tl.trans(v), allow_tf32=True)    # dP = dO @ V^T
        ds = p * (dp - Di[:, None])                       # dS = P * (dP - D)

        dq_acc += sm_scale * tl.dot(ds, k, allow_tf32=True)  # dQ += scale * dS @ K

    tl.store(dQ_ptr + pid_bh * sdqb + offs_q[:, None] * sdqt + offs_d[None, :] * sdqd,
             dq_acc.to(tl.float16), mask=mask_q)


@triton.jit
def _bwd_dkdv_kernel(
    Q_ptr, K_ptr, V_ptr,
    L_ptr, D_ptr, dO_ptr,
    dK_ptr, dV_ptr,
    k_row_offsets_ptr, k_col_indices_ptr,
    sm_scale,
    sqb, sqt, sqd,
    skb, skt, skd,
    svb, svt, svd,
    slb, slt,
    sdb, sdt,
    sdob, sdot, sdod,
    sdkb, sdkt, sdkd,
    sdvb, sdvt, sdvd,
    T, d: tl.constexpr,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
    LOG2E: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_k  = tl.program_id(1)  # 第几个key block

    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, d)
    mask_k = offs_k[:, None] < T

    k = tl.load(K_ptr + pid_bh * skb + offs_k[:, None] * skt + offs_d[None, :] * skd,
                mask=mask_k, other=0.0).to(tl.float32)
    v = tl.load(V_ptr + pid_bh * svb + offs_k[:, None] * svt + offs_d[None, :] * svd,
                mask=mask_k, other=0.0).to(tl.float32)

    dk_acc = tl.zeros((BLOCK_K, d), dtype=tl.float32)
    dv_acc = tl.zeros((BLOCK_K, d), dtype=tl.float32)

    lo = tl.load(k_row_offsets_ptr + pid_k)
    hi = tl.load(k_row_offsets_ptr + pid_k + 1)

    for idx in range(lo, hi):
        q_block = tl.load(k_col_indices_ptr + idx)  # 哪些query block attend了这个key block
        offs_q = q_block * BLOCK_Q + tl.arange(0, BLOCK_Q)
        mask_q = offs_q[:, None] < T

        qi  = tl.load(Q_ptr  + pid_bh * sqb  + offs_q[:, None] * sqt  + offs_d[None, :] * sqd,
                      mask=mask_q, other=0.0).to(tl.float32)
        doi = tl.load(dO_ptr + pid_bh * sdob + offs_q[:, None] * sdot + offs_d[None, :] * sdod,
                      mask=mask_q, other=0.0).to(tl.float32)
        Li  = tl.load(L_ptr  + pid_bh * slb  + offs_q * slt, mask=offs_q < T, other=0.0)
        Di  = tl.load(D_ptr  + pid_bh * sdb  + offs_q * sdt, mask=offs_q < T, other=0.0)

        # 重新算P
        scores = tl.dot(qi, tl.trans(k), allow_tf32=True) * sm_scale
        scores = tl.where(offs_k[None, :] < T, scores, float('-inf'))
        p = tl.exp2(scores * LOG2E - Li[:, None])  # (BLOCK_Q, BLOCK_K)

        dv_acc += tl.dot(tl.trans(p), doi, allow_tf32=True)  # dV += P^T @ dO

        dp = tl.dot(doi, tl.trans(v), allow_tf32=True)
        ds = p * (dp - Di[:, None])

        dk_acc += sm_scale * tl.dot(tl.trans(ds), qi, allow_tf32=True)  # dK += scale * dS^T @ Q

    tl.store(dK_ptr + pid_bh * sdkb + offs_k[:, None] * sdkt + offs_d[None, :] * sdkd,
             dk_acc.to(tl.float16), mask=mask_k)
    tl.store(dV_ptr + pid_bh * sdvb + offs_k[:, None] * sdvt + offs_d[None, :] * sdvd,
             dv_acc.to(tl.float16), mask=mask_k)


def sparse_flash_backward(Q, K, V, O, L, dO,
                          k_row_offsets, k_col_indices,
                          q_row_offsets, q_col_indices,
                          sm_scale, BLOCK_Q, BLOCK_K):
    B, H, T, d = Q.shape
    BH = B * H
    LOG2E = 1.4426950408889634

    dQ = torch.empty_like(Q)
    dK = torch.empty_like(K)
    dV = torch.empty_like(V)
    D  = torch.empty(B, H, T, device=Q.device, dtype=torch.float32)  # 存D_i用

    nq = T // BLOCK_Q
    nk = T // BLOCK_K

    # 第一步：算D_i = rowsum(dO * O)
    _bwd_preprocess[(BH, nq)](
        O, dO, D,
        O.stride(1), O.stride(2), O.stride(3),
        dO.stride(1), dO.stride(2), dO.stride(3),
        D.stride(1), D.stride(2),
        T, d, BLOCK_Q=BLOCK_Q,
        num_warps=4,
    )

    # 第二步：算dQ，每个query block独立，iterate over它的live key blocks
    _bwd_dq_kernel[(BH, nq)](
        Q, K, V, L, D, dO, dQ,
        q_row_offsets, q_col_indices,
        sm_scale,
        Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(1), K.stride(2), K.stride(3),
        V.stride(1), V.stride(2), V.stride(3),
        L.stride(1), L.stride(2),
        D.stride(1), D.stride(2),
        dO.stride(1), dO.stride(2), dO.stride(3),
        dQ.stride(1), dQ.stride(2), dQ.stride(3),
        T, d,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K,
        LOG2E=LOG2E,
        num_warps=4,
    )

    # 第三步：算dK和dV，每个key block独立，iterate over attend它的query blocks
    _bwd_dkdv_kernel[(BH, nk)](
        Q, K, V, L, D, dO, dK, dV,
        k_row_offsets, k_col_indices,
        sm_scale,
        Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(1), K.stride(2), K.stride(3),
        V.stride(1), V.stride(2), V.stride(3),
        L.stride(1), L.stride(2),
        D.stride(1), D.stride(2),
        dO.stride(1), dO.stride(2), dO.stride(3),
        dK.stride(1), dK.stride(2), dK.stride(3),
        dV.stride(1), dV.stride(2), dV.stride(3),
        T, d,
        BLOCK_Q=BLOCK_Q, BLOCK_K=BLOCK_K,
        LOG2E=LOG2E,
        num_warps=4,
    )

    return dQ, dK, dV