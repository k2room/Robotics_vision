import cv2
import numpy as np

# ONNX 모델 불러오기
net = cv2.dnn.readNetFromONNX("model.onnx")

# 입력 데이터 전처리
# 예시: 이미지를 로드하고 전처리 (크기 조정, 정규화 등)
image = cv2.imread("image.jpg")
blob = cv2.dnn.blobFromImage(image, 1/255.0, (224, 224), (0, 0, 0), swapRB=True, crop=False)

# 모델에 입력 데이터 설정
net.setInput(blob)

# 추론 수행
output = net.forward()

# 결과 처리 (예: 분류 결과 출력)
print(output)
