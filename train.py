import torch
import torch.onnx

# 모델 정의 및 로드
model = YourModel()
model.load_state_dict(torch.load("model.pth"))

# 평가 모드로 설정
model.eval()

# 더미 입력 데이터 생성 (예: 1x3x224x224 크기의 이미지)
dummy_input = torch.randn(1, 3, 224, 224)

# ONNX로 내보내기
torch.onnx.export(model, dummy_input, "model.onnx")
