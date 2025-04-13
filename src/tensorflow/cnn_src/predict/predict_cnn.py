import numpy as np
import librosa
from tensorflow.keras.models import load_model
import subprocess
import os
import uuid
import sys

# 📌 설정값
sr = 22050
n_mfcc = 13
hop_length = 512
max_len = 130
model_path = "cnn_model.h5"

# ✅ 모델 불러오기
model = load_model(model_path)

def plot_results(results):
    if not results:
        print("❗ 시각화할 데이터가 없어요!")
        return

def convert_to_wav(file_path):
    """mp4/m4a/mp3 → wav로 변환 (임시 파일 생성)"""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.wav':
        return file_path

    temp_wav = f"temp_{uuid.uuid4().hex[:8]}.wav"
    command = ['ffmpeg', '-y', '-i', file_path, '-ac', '1', '-ar', str(sr), temp_wav]

    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return temp_wav
    except subprocess.CalledProcessError:
        print("❗ ffmpeg 변환 실패!")
        return None

def preprocess_audio(file_path):
    """오디오 파일을 CNN 입력용 형태로 전처리"""
    y_audio, _ = librosa.load(file_path, sr=sr, duration=5.0)

    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
    features = np.vstack([mfcc, zcr])  # (14, N)

    if features.shape[1] < max_len:
        pad_width = max_len - features.shape[1]
        features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
    else:
        features = features[:, :max_len]

    features = features[..., np.newaxis]  # (14, 130, 1)
    return np.expand_dims(features, axis=0)  # (1, 14, 130, 1)

def predict_audio(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    temp_file = None

    if ext in ['.mp4', '.m4a', '.mp3']:
        print(f"🔄 변환 중: {file_path} → wav")
        temp_file = convert_to_wav(file_path)
        if temp_file is None:
            return
        wav_path = temp_file
    else:
        wav_path = file_path

    X_new = preprocess_audio(wav_path)
    prediction = model.predict(X_new)[0][0]

    print(f"\n🔍 예측 확률: {prediction:.4f}")
    print("✅ 분류 결과:", "🔈 조용한 소리 (quiet)" if prediction < 0.5 else "📢 시끄러운 소리 (loud)")

    if temp_file and os.path.exists(temp_file):
        os.remove(temp_file)

# ✅ 실행 진입점
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❗ 사용법: python predict_cnn.py [파일명]")
    else:
        test_file = sys.argv[1]
        if not os.path.exists(test_file):
            print(f"❗ 오류: 파일 '{test_file}' 이 존재하지 않아요.")
        else:
            predict_audio(test_file)
