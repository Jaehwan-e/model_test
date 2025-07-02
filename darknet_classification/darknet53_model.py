import torch
import torch.nn as nn

# Residual Block 정의 (1x1 conv + 3x3 conv)
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        mid_channels = in_channels // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv2d(mid_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return x + self.block(x)

# Darknet-53 전체 구조
class Darknet53(nn.Module):
    def __init__(self, num_classes=1000):
        super(Darknet53, self).__init__()
        def conv_bn_lrelu(in_channels, out_channels, kernel_size, stride):
            pad = kernel_size // 2
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.1, inplace=True)
            )

        self.stem = conv_bn_lrelu(3, 32, 3, 1)
        self.stage1 = self._make_stage(32, 64, 1)
        self.stage2 = self._make_stage(64, 128, 2)
        self.stage3 = self._make_stage(128, 256, 8)
        self.stage4 = self._make_stage(256, 512, 8)
        self.stage5 = self._make_stage(512, 1024, 4)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, num_classes)

    def _make_stage(self, in_channels, out_channels, num_blocks):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        ]
        for _ in range(num_blocks):
            layers.append(ResidualBlock(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)        # 1. 초기 conv 처리
        x = self.stage1(x)      # 2. 다운샘플링 + residual 1회
        x = self.stage2(x)      # 3. 다운샘플링 + residual 2회
        x = self.stage3(x)      # 4. 다운샘플링 + residual 8회 ← feat1
        x = self.stage4(x)      # 5. 다운샘플링 + residual 8회 ← feat2
        x = self.stage5(x)      # 6. 다운샘플링 + residual 4회 ← feat3
        x = self.global_pool(x) # 7. Adaptive Average Pooling
        x = x.view(x.size(0), -1) # 8. 1D 벡터로 변환
        return self.fc(x)       # 9. 최종 fully-connected 분류