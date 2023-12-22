import torch
from torch import empty
from torch._dynamo.testing import rand_strided
import triton
import triton.language as tl


@triton.jit
def triton_(
    in_ptr0,
    in_ptr1,
    in_ptr2,
    two_by,
    one_by,
    out_ptr0,
    out_ptr1,
    XBLOCK: tl.constexpr,
    VERSION: tl.constexpr,
):
    xnumel = 1024
    rnumel = 128
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex_ = xoffset + tl.arange(0, XBLOCK)
    xindex = xindex_[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex % 16
    r2 = rindex // 16
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (16 * x0) + (16384 * r2)), rmask & xmask).to(tl.int1)
    tmp1 = tl.load(in_ptr1 + (0))
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp4 = tl.load(
        in_ptr2 + (x0 + (1024 * r2)),
        rmask & xmask,
        eviction_policy="evict_last",
        other=0.0,
    ).to(tl.float32)
    tmp9 = tl.load(two_by + (r1 + (16 * x0) + (16384 * r2)), rmask & xmask, other=0.0)

    if VERSION == 0:
        # works loading with a 2D [XBLOCK, 1] block
        tmp10 = tl.load(one_by + (x0), xmask)
    elif VERSION == 2:
        # tl.reshape() causes a RuntimeError: Triton Error [CUDA]: misaligned address
        index_1d = xoffset + tl.arange(0, XBLOCK)
        tmp10 = tl.reshape(
            tl.load(one_by + index_1d, index_1d < xnumel),
            [XBLOCK, 1],
        )

    # removing any of these breaks the bug
    tmp7 = tl.where(tmp0, tmp2, tmp4)
    tmp11 = tmp9 - tmp10

    # maybe here?
    tmp12 = tmp7 * tmp11
    
    tmp16 = tl.sum(tmp12, 1)[:, None]
    tmp17 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp16, xmask)
    tl.store(out_ptr1 + (x0), tmp20, xmask)


arg0_1 = rand_strided(
    (8, 1024, 4, 4), (1024, 1, 0, 0), device="cuda:0", dtype=torch.float16
)
arg1_1 = rand_strided(
    (8, 1024, 4, 4), (16384, 16, 4, 1), device="cuda:0", dtype=torch.bool
)
arg2_1 = rand_strided((), (), device="cuda:0", dtype=torch.float32)
arg3_1 = rand_strided(
    (8, 1024, 4, 4), (16384, 16, 4, 1), device="cuda:0", dtype=torch.float32
)
arg4_1 = rand_strided(
    (1, 1024, 1, 1), (1024, 1, 1, 1), device="cuda:0", dtype=torch.float32
)
buf0 = empty((1024,), device="cuda", dtype=torch.float32)
buf1 = empty((1024,), device="cuda", dtype=torch.float32)
xblock = 1

print(triton.compile(triton_), signature="*f32,*f32")
print(buf1)
triton_[(triton.cdiv(1024, xblock),)](
    arg1_1,
    arg2_1,
    arg0_1,
    arg3_1,
    arg4_1,
    buf0,
    buf1,
    XBLOCK=xblock,
    VERSION=2,
)
print(buf1)
torch.cuda.synchronize()
print("SUCCESS")