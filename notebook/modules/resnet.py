import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(BasicBlock, self).__init__()

        # 1.합성곱층 정의
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1)

        self.downsample = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # 2.배치 정규화층 정의
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.relu = nn.ReLU()
        
    def forward(self, x):
        # 3.스킵 커넥션을 위해 초기 입력을 저장
        x_ = x

        x = self.c1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.c2(x)
        x = self.bn2(x)

        # 4.합성곱의 결과와 입력의 채널 수를 맞춤
        x_ = self.downsample(x_)

        # 5.합성곱층의 결과와 저장해놨던 입력값을 더해줌
        x += x_
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()

        #Output Size = (W - F + 2P) / S + 1
        #W: input_volume_size
        #F: kernel_size
        #P: padding_size
        #S: strides
        #nn.Conv2d(in_channels=3, out_channels=32, kernel=3, padding=1)
        #output_size = (32 - 3 + 2*1) / 1 + 1 = 32

        # ❶ 기본 블록
        # 기본 입력이 RGB인 3채널
        self.b1 = BasicBlock(in_channels=3, out_channels=64)
        self.b2 = BasicBlock(in_channels=64, out_channels=128)
        self.b3 = BasicBlock(in_channels=128, out_channels=256)


        # ❷ 풀링을 최댓값이 아닌 평균값으로
        # maxpooling=2
        #input_filter_size/2 = output_filter_size
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # ❸ 분류기  MLP
        # 4 * 4* 256
        # out_feature * kerner
        # (batchsize/2) * imgsize
        # 4 * 224 * 224
        #self.fc1 = nn.Linear(in_features= 4 * 128 * 128, out_features=2048)
        self.fc1 = nn.Linear(in_features=4 * 64 * 64, out_features=2048)
        self.fc2 = nn.Linear(in_features=2048, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

        self.relu = nn.ReLU()
    def forward(self, x):
        
        # 1. 기본 블록과 풀링층을 통과
        x = self.b1(x)
        x = self.pool(x)
        x = self.b2(x)
        x = self.pool(x)
        x = self.b3(x)
        x = self.pool(x)


        # ❷ 분류기의 입력으로 사용하기 위해 flatten
        x = torch.flatten(x, start_dim=1)

        # ❸ 분류기로 예측값 출력
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x