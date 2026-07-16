import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        concat_channels = out_channels + skip_channels
        self.conv1 = nn.Conv2d(concat_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class UNetResNet(nn.Module):
    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = resnet34(weights=weights)

        if in_channels != 3:
            old_conv = resnet.conv1
            new_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    new_conv.weight[:, :3] = old_conv.weight
                    for c in range(3, in_channels):
                        new_conv.weight[:, c] = old_conv.weight.mean(dim=1)
            resnet.conv1 = new_conv

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  
        self.layer2 = resnet.layer2  
        self.layer3 = resnet.layer3  
        self.layer4 = resnet.layer4  

        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 64)

        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.relu(self.bn1(self.conv1(x)))          
        x_pool = self.maxpool(x0)                        
        x1 = self.layer1(x_pool)                         
        x2 = self.layer2(x1)                             
        x3 = self.layer3(x2)                             
        x4 = self.layer4(x3)                             

        d4 = self.dec4(x4, x3)   
        d3 = self.dec3(d4, x2)   
        d2 = self.dec2(d3, x1)   
        d1 = self.dec1(d2, x0)   

        out = self.final_upsample(d1)   
        out = self.final_conv(out)      
        out = self.sigmoid(out)         
        return out

# ------------------------------------------------------------------------------
# 3. Dice Loss
# ------------------------------------------------------------------------------
class DiceLoss(nn.Module):
    """
    Dice loss for binary segmentation.
    Dice = 2 * |P ∩ G| / (|P| + |G|)
    Loss = 1 - Dice
    """

    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: Predicted probabilities of shape (B, 1, H, W).
            targets: Ground truth binary masks of same shape.
        Returns:
            Scalar loss (mean over batch).
        """
        # Flatten spatial dimensions
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        loss = 1.0 - dice
        return loss.mean()


# ------------------------------------------------------------------------------
# 4. Hybrid Road Extraction Loss (BCE + Dice)
# ------------------------------------------------------------------------------
class RoadExtractionLoss(nn.Module):
    """
    Hybrid loss combining Binary Cross-Entropy (BCE) and Dice loss
    to handle heavy class imbalance (roads < 5% of pixels).
    Loss = 0.5 * BCE + 0.5 * Dice
    """

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5):
        """
        Args:
            dice_weight: Weight for Dice loss.
            bce_weight: Weight for BCE loss.
        """
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def forward(self, probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            probs: Predicted probabilities (B, 1, H, W).
            targets: Ground truth masks (B, 1, H, W).
        Returns:
            Combined scalar loss.
        """
        bce = self.bce_loss(probs, targets)
        dice = self.dice_loss(probs, targets)
        return self.bce_weight * bce + self.dice_weight * dice


# ------------------------------------------------------------------------------
# 5. IoU Metric
# ------------------------------------------------------------------------------
def calculate_iou(
    probs: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """
    Compute the Intersection over Union (IoU) for a batch of predictions.

    Args:
        probs: Predicted probabilities of shape (B, 1, H, W).
        targets: Ground truth binary masks of same shape.
        threshold: Decision threshold for converting probabilities to binary.
        smooth: Small constant for numerical stability.

    Returns:
        Mean IoU over the batch as a 0‑dim tensor.
    """
    # Binarise predictions
    preds = (probs > threshold).float()
    # Intersection and union summed over spatial dimensions
    intersection = (preds * targets).sum(dim=(2, 3))  # (B, 1)
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


# ------------------------------------------------------------------------------
# 6. Validation / Test Block
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Instantiate the model with RGB input
    model = UNetResNet(in_channels=3, pretrained=True).to(device)
    model.eval()

    # Dummy input: batch of 2 satellite images (512x512, 3 channels)
    dummy_input = torch.randn(2, 3, 512, 512).to(device)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Input shape : {dummy_input.shape}")
    print(f"Output shape: {output.shape}")   # Expected: [2, 1, 512, 512]
    print(f"Output value range: [{output.min().item():.3f}, {output.max().item():.3f}]")

    # Quick test of loss and metric with random targets
    dummy_targets = torch.randint(0, 2, (2, 1, 512, 512), dtype=torch.float32).to(device)
    loss_fn = RoadExtractionLoss()
    loss_val = loss_fn(output, dummy_targets)
    iou_val = calculate_iou(output, dummy_targets)

    print(f"Dummy hybrid loss: {loss_val.item():.4f}")
    print(f"Dummy mean IoU  : {iou_val.item():.4f}")

    # Additional test for 4-channel input (RGB+NIR)
    model_nir = UNetResNet(in_channels=4, pretrained=True).to(device)
    model_nir.eval()
    dummy_input_nir = torch.randn(1, 4, 512, 512).to(device)
    with torch.no_grad():
        output_nir = model_nir(dummy_input_nir)
    print(f"\n4-channel test output shape: {output_nir.shape}")  # [1, 1, 512, 512]
