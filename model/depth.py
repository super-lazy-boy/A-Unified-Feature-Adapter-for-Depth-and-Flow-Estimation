from torch import nn
class DepthHead(nn.Module):
    """
    A lightweight refinement head for depth prediction.
    Input : coarse depth from DepthAnythingV2, shape [B, 1, h, w]
    Output: refined depth, shape [B, 1, h, w]
    """
    def __init__(self, in_ch=1, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1, padding=0),
        )

    def forward(self, x):
        return self.net(x)
