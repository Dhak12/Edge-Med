import torch
import monai
from monai.networks.nets import DenseNet121

model = DenseNet121(
    spatial_dims=2,
    in_channels=1,
    out_channels=3
)

x = torch.randn(1, 1, 224, 224)
y = model(x)

print("MONAI:", monai.__version__)
print("PyTorch:", torch.__version__)
print("Output shape:", y.shape)