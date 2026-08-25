import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 1. 테스트용 데이터셋 및 데이터로더 준비
# (만약 테스트셋 폴더가 따로 있다면 경로를 수정하시고, 없다면 TrainSet을 그대로 넣어서 확인해볼 수도 있습니다)
test_dir = '/content/drive/MyDrive/kuffers/kuffers/TestSet' # 혹은 TestSet 경로

test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
# 셔플을 False로 해야 이미지와 예측 결과를 나란히 확인할 때 좋습니다.
test_loader = DataLoader(test_dataset, batch_size=15, shuffle=False) 

# 2. 저장해 둔 모델 불러오기
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 아까 정의했던 모델 구조를 다시 메모리에 올립니다.
model = PillInspectionCNN().to(device)
# 저장했던 가중치(.pth)를 불러옵니다.
model.load_state_dict(torch.load('/content/drive/MyDrive/pill_model.pth'))
model.eval() # 평가 모드로 설정

# 클래스 이름 매핑 (0: NG, 1: OK - 아까 출력된 순서 기준)
classes = ['NG', 'OK']

# 3. 데이터 한 배치(15장)를 꺼내서 예측하고 시각화하기
dataiter = iter(test_loader)
images, labels = next(dataiter)

images = images.to(device)
labels = labels.to(device)

# 모델에 사진 넣어서 예측값 뽑기
outputs = model(images)
_, preds = torch.max(outputs, 1)

# 4. 결과 시각화 (matplotlib 활용)
fig = plt.figure(figsize=(12, 8))

for idx in range(15):
    plt.subplot(3, 5, idx + 1)
    
    # 텐서 이미지를 matplotlib 형식(H, W, C)으로 변환
    img = images[idx].cpu().numpy().transpose((1, 2, 0))
    plt.imshow(img)
    
    pred_name = classes[preds[idx].item()]
    true_name = classes[labels[idx].item()]
    
    # 예측 결과가 맞으면 파란색 글씨, 틀리면 빨간색 글씨로 표시
    color = 'blue' if pred_name == true_name else 'red'
    
    plt.title(f"Pred: {pred_name}\nTrue: {true_name}", color=color, fontsize=12)
    plt.axis('off')

plt.tight_layout()
plt.show()