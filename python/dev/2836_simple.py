import torch
from torch import empty
from torch._dynamo.testing import rand_strided
import triton
import triton.language as tl

@triton.jit
def working_(
    in_ptr,
    out_ptr
):
    xnumel = 1024

    xoffset = tl.program_id(0)
    xindex = xoffset + tl.arange(0, 1)[:, None]

    index_1d = xoffset + tl.arange(0, 1)
    out_tensor = tl.load(in_ptr + index_1d, index_1d < xnumel)[:, None]
    tl.store(out_ptr + xindex, out_tensor)


@triton.jit
def broken_(
   in_ptr,
   out_ptr_one,
   out_ptr_two
): 
    XBLOCK: tl.constexpr = 1
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, 1)[:, None]
    x0 = xindex
    

    
    RBLOCK: tl.constexpr = 128
    rindex = tl.arange(0, RBLOCK)[None, :]
    r1 = rindex % 16
    r2 = rindex // 16

    rnumel = 128


    # tl.reshape() causes a RuntimeError: Triton Error [CUDA]: misaligned address
    index_1d = xoffset + tl.arange(0, XBLOCK)
    tmp10 = tl.reshape(
        tl.load(in_ptr + index_1d, index_1d < xnumel),
        [XBLOCK, 1],
    )

    tl.store(out_ptr_one + xindex, tmp10)
    tl.store(out_ptr_two + xindex, tmp10)



in_tensor = rand_strided(
    (1, 1024, 1, 1), (1024, 1, 1, 1), device="cuda:0", dtype=torch.float32
)
in_tensor_two = rand_strided(
    (8, 1024, 4, 4), (16384, 16, 4, 1), device="cuda:0", dtype=torch.float32
)

out_buf1 = empty((1024,), device="cuda", dtype=torch.float32)
out_buf2 = empty((1024,), device="cuda", dtype=torch.float32)


broken_[(1024,)](
    in_tensor,
    out_buf1,
    out_buf2
)

torch.cuda.synchronize()
print("SUCCESS")