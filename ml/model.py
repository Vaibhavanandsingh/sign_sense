from torch import nn


def conv_block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Dropout(dropout),
    )


class SignNet(nn.Module):
    def __init__(self, num_classes: int = 43, img_size: int = 48):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 32, 0.1),
            conv_block(32, 64, 0.2),
            conv_block(64, 128, 0.3),
        )
        feature_size = img_size // 8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feature_size * feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
