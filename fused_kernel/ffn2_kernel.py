# os.environ["TRITON_INTERPRET"] = "1"
import triton
import triton.language as tl

# why difference is huge in INTERPRET mode, issue: 
@triton.jit
def ffn2_kernel(input_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, 
                output_ptr,
                d1: tl.constexpr, d2: tl.constexpr, d3: tl.constexpr,
                bs1: tl.constexpr, bs2: tl.constexpr, bs3: tl.constexpr, BLOCK_SIZE: tl.constexpr
               ):
    pid_d1 = tl.program_id(0)
    pid_d3 = tl.program_id(1)

    offsets_d1 = (tl.arange(0, bs1) + bs1 * pid_d1)
    offsets_d3 = (tl.arange(0, bs3) + bs3 * pid_d3)
    t_val = tl.zeros((bs1, bs3), dtype=tl.float32)
    input_ptrs = input_ptr + (offsets_d1[:, None] * d2 + tl.arange(0, BLOCK_SIZE)[None, :])
    w1_ptrs = w1_ptr + (offsets_d3[:, None] * d2 + tl.arange(0, BLOCK_SIZE)[None, :])
    for index in range(0, tl.cdiv(d2, BLOCK_SIZE)):
        mask_input = (offsets_d1[:, None] < d1) & (tl.arange(0, BLOCK_SIZE)[None, :] < d2 - index * BLOCK_SIZE)
        input_val = tl.load(input_ptrs, mask = mask_input, other = 0.0)

        mask_w1 = (offsets_d3[:, None] < d3) & (tl.arange(0, BLOCK_SIZE)[None, :] < d2 - index * BLOCK_SIZE)
        w1_val = tl.load(w1_ptrs, mask = mask_w1, other = 0.0)
        w1_val = w1_val.trans(1, 0)

        t_val += tl.dot(input_val, w1_val)

        input_ptrs += BLOCK_SIZE
        w1_ptrs += BLOCK_SIZE
        
    w2_ptrs = w2_ptr + tl.arange(0, BLOCK_SIZE)[:, None] * d3 + (tl.arange(0, bs3) + pid_d3 * bs3)[None, :]
    output_ptrs = output_ptr + offsets_d1[:, None] * d2 + tl.arange(0, BLOCK_SIZE)[None, :]
    for index in range(0, tl.cdiv(d2, BLOCK_SIZE)):
        mask_w2 = (tl.arange(0, BLOCK_SIZE)[:, None] < d2 - index * BLOCK_SIZE) & (tl.arange(0, bs3)[None, :] < d3 - pid_d3 * bs3)
        w2_val = tl.load(w2_ptrs, mask = mask_w2, other = 0.0)
        w2_val = w2_val.trans(1, 0)
        output_val = tl.dot(t_val, w2_val)

        mask_output = (offsets_d1[:, None] < d1) & (tl.arange(0, BLOCK_SIZE)[None, :] < d2 - index * BLOCK_SIZE)
        tl.atomic_add(output_ptrs, val = output_val, mask = mask_output)

        w2_ptrs += BLOCK_SIZE * d3
        output_ptrs += BLOCK_SIZE
