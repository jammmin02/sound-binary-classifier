# 베이스 이미지: PyTorch + CUDA 12.1
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# ffmpeg 설치 (오디오 변환용)
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 작업 디렉토리
WORKDIR /app

# 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 프로젝트 전체 복사
COPY . .

# 세 개의 Python 스크립트 자동 실행
CMD ["bash", "-c", "python pytorch_src/preprocessing.py && python pytorch_src/train.py && python pytorch_src/predict.py"]
