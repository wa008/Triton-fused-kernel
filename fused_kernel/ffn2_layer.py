import torch
# os.environ["TRITON_INTERPRET"] = "1"
import triton
import triton.language as tl

# fused nn.Linear2(nn.Linear1(input))
class Linear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int, 
        bias: bool = True,
        device: Device = 'cuda',
        dtype: torch.dtype = torch.float32,
        block_size
    ) -> None:
    '''
        l1: out_features -> hidden_features
        l2: hidden_features -> out_features 
    '''
        l1 = nn.Linear(out_features, hidden_features, bias, device, dtype)
        l2 = nn.Linear(hidden_features, out_features, bias, device, dtype)
        if block_size is not None:
            self.block_size = block_size

    @staticmethod
    def forward(self, input: Tensor) -> Tensor:
        '''
            input: d1 x d2
            output: d1 x d2
        '''
        (d1, d2) = input.shape
        d3 = l1.weight.shape[0]
        output = torch.zeros(input.shape, dtype = input.dtype, device = input.device)
        grid = lambda META: (triton.cdiv(d1, self.block_size), triton.cdiv(d3, self.block_size))
        ffn2_kernel[grid](input, self.l1.weight, self.l1.bias, self.l2.weight, self.l2.bias,
                            output,
                            d1, d2, d3,
                            self.block_size, self.block_size, self.block_size, BLOCK_SIZE = self.block_size)
        return output

    @staticmethod
    def backward():
        pass