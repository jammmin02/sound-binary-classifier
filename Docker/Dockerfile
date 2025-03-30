# TensorFlow 학습용 Python 이미지 기반
FROM python:3.10-slim

# 시스템 패키지 설치 (ffmpeg 포함)
RUN apt-get update && apt-get install -y ffmpeg git && apt-get clean

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 코드 복사
COPY . .

# 실행 명령 (예: CNN 데이터셋 만들기 → 학습)
CMD ["python", "cnn_train.py"]
