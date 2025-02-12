import triton
import triton.language as tl

@triton.jit
def fast_cross_entropy_loss_kernel(input_ptr,
                input_grad_ptr,
                target_ptr,
                output_ptr,
                M, N,
                BLOCK_SIZE_N: tl.constexpr,
               ):
    pid_m = tl.program_id(0)
    input_ptr += pid_m * N
    target_ptr += pid_m
    output_ptr += pid_m
    offsets_cols = tl.arange(0, BLOCK_SIZE_N)

    target = tl.load(target_ptr)
    max_val = -float("inf")
    sumexp = 0.0
    allcurx = 0.0

    for index in tl.range(0, N, BLOCK_SIZE_N):
        offsets_input = (offsets_cols + index)
        mask_input = (offsets_cols + index) < N
        input_val = tl.load(input_ptr + offsets_input, mask=mask_input, other = -float("inf"))

        if index == 0:
            new_max_val = tl.max(input_val)
        else:
            new_max_val = tl.maximum(max_val, tl.max(input_val))

        sumexp /= tl.exp(new_max_val)
        sumexp *= tl.exp(max_val)
        sumexp += tl.sum(tl.exp(input_val - new_max_val))
        max_val = new_max_val
        if sumexp > 1.0:
            max_val -= tl.log(1.0 / sumexp)
            sumexp = 1.0

        if (target >= index) and (target < index + BLOCK_SIZE_N):
            curx = tl.load(input_ptr + target)
            allcurx += curx
    for index in tl.range(0, N, BLOCK_SIZE_N):
        offsets_input = (offsets_cols + index)
        mask_input = (offsets_cols + index) < N
        input_val = tl.load(input_ptr + offsets_input, mask=mask_input, other = -float("inf"))
        target_1d = (offsets_cols + index)
        grad = tl.div_rn(tl.exp(input_val - max_val), sumexp) - tl.where(target_1d == target, 1, 0)
        tl.store(input_grad_ptr + offsets_input, grad, mask = mask_input)
    output = tl.log(sumexp) + max_val - allcurx
    tl.store(output_ptr, output)