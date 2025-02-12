class FastCrossEntropyLossAutoGrad(torch.autograd.Function):
    @staticmethod
    def forward(input: torch.Tensor, target: torch.Tensor, bm = 1, bn = 256):
        output = torch.empty(target.shape, dtype = input.dtype, device = input.device)
        grid_loss = (input.shape[0], )
        fast_cross_entropy_loss_kernel[grid_loss](input, input.grad, target, output, M = input.shape[0], N = input.shape[1], BLOCK_SIZE_N = bn)
        return output

class FastCrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(
        self,
        reduction: str = 'mean',
        size_average: Optional[bool] = None,
        reduce: Optional[bool] = None,
        weight: Optional[Tensor] = None,
        ignore_index: int = -100,
        label_smoothing: float = 0.0,
        ) -> None:
        super().__init__(weight, size_average, ignore_index, reduce,
                         reduction, label_smoothing)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return FastCrossEntropyLossAutoGrad.apply(input, target)
