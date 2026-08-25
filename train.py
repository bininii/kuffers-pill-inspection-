import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 1. 사용할 장치 설정 (GPU가 있으면 GPU, 없으면 CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용중인 장치: {device}")

# 2. 데이터 전처리 및 데이터로더 설정 (구글 드라이브 경로 연결!)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# 구글 드라이브에 업로드한 경로로 설정
train_dir = '/content/drive/MyDrive/kuffers/kuffers/TrainSet'
test_dir = '/content/drive/MyDrive/kuffers/TestSet'  # 테스트셋 폴더가 따로 있다면 이 경로로 설정

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
print("클래스 정보:", train_dataset.class_to_idx)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


# 3. CNN 모델 구조 정의 (알약 검사용)
class PillInspectionCNN(nn.Module):
    def __init__(self):
        super(PillInspectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 2)  # 출력 2개: OK, NG
        )

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.fc(x)
        return x

model = PillInspectionCNN().to(device)


# 4. 손실 함수와 최적화 함수 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 5. 모델 학습 실행 (Epoch은 5번 반복)
epochs = 5
print("학습을 시작합니다...")

for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"[{epoch}/{epochs}] 학습 손실(Loss): {running_loss / len(train_loader):.4f}")

# 모델 저장하기
torch.save(model.state_dict(), '/content/drive/MyDrive/pill_model.pth')
print("학습 완료! 구글 드라이브에 모델이 저장되었습니다.")