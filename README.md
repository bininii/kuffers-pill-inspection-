# 💊 CNN 기반 알약 불량 검출 (Kuffers Image Classification)

컴퓨터 비전(Computer Vision) 기술을 활용하여 제조 공정 내 알약 이미지의 정상(OK)과 불량(NG)을 자동으로 판별하는 딥러닝 분류 모델입니다.

## 🚀 프로젝트 개요
- **목적**: 제조업 스마트 팩토리 환경을 가정하여 이미지 기반 양품/불량 자동 분류 파이프라인 구축
- **사용 기술**: Python, PyTorch, TorchVision, Matplotlib
- **모델 구조**: 커스텀 CNN (3개의 Conv 블록 + Fully Connected Layer)

## 📂 데이터셋 (Dataset)
- **Kuffers Dataset**: 총 수십 장의 알약 접시 이미지를 활용
- **클래스 구성**: 
  - `OK`: 정상 제품 (0)
  - `NG`: 불량 제품 (1)

## 🛠️ 파이프라인 구조
1. **Data Pipeline**: `ImageFolder`와 `DataLoader`를 활용한 배치 단위 이미지 전처리 및 로드 (`Resize 64x64`, `ToTensor`)
2. **Model Architecture**: 2개 클래스 분류를 위한 합성곱 신경망(CNN) 정의
3. **Training & Evaluation**: CrossEntropyLoss와 Adam 옵티마이저를 활용한 학습 및 정확도 측정
4. **Inference & Visualization**: 학습된 `.pth` 모델을 불러와 테스트 이미지에 대한 예측 결과(`Pred`)와 정답(`True`)을 시각화

## 📊 실행 결과 예시
- 모델 학습 완료 후 테스트 세트에 대한 예측 정확도(Accuracy) 산출 및 시각화 완료
