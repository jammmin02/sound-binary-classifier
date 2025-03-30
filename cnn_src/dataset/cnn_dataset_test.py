import os
import librosa
import numpy as np
import subprocess
import matplotlib.pyplot as plt

# 설정
base_dir = 'data'                          # 오디오 파일 폴더
output_dir = 'outputs/cnn'                 # CNN 결과 저장 폴더
sr = 22050
n_mfcc = 13
hop_length = 512
max_len = 130
segment_duration = 5.0

X = []
y = []

def convert_mp4_to_wav(mp4_path, wav_path):
    if not os.path.exists(wav_path):
        command = ['ffmpeg', '-y', '-i', mp4_path, '-ac', '1', '-ar', str(sr), wav_path]
        subprocess.run(command, check=True)

# 폴더 생성
for label_name in ['quiet', 'loud']:
    folder_path = os.path.join(base_dir, label_name)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visuals', label_name), exist_ok=True)

# 데이터 처리
for label_name in ['quiet', 'loud']:
    folder_path = os.path.join(base_dir, label_name)
    label = 0 if label_name == 'quiet' else 1

    for file_name in os.listdir(folder_path):
        if not file_name.endswith(('.wav', '.mp4', '.m4a')):
            continue

        file_path = os.path.join(folder_path, file_name)

        if file_name.endswith(('.mp4', '.m4a')):
            wav_name = os.path.splitext(file_name)[0] + '.wav'
            wav_path = os.path.join(folder_path, wav_name)
            convert_mp4_to_wav(file_path, wav_path)
            file_path = wav_path

        base_filename = os.path.splitext(file_name)[0]
        total_duration = librosa.get_duration(path=file_path)
        segment_count = int(total_duration // segment_duration)

        for i in range(segment_count):
            offset = i * segment_duration
            y_audio, _ = librosa.load(file_path, sr=sr, offset=offset, duration=segment_duration)

            mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
            zcr = librosa.feature.zero_crossing_rate(y=y_audio, hop_length=hop_length)
            features = np.vstack([mfcc, zcr])  # (14, N)

            if features.shape[1] < max_len:
                pad_width = max_len - features.shape[1]
                features = np.pad(features, ((0, 0), (0, pad_width)), mode='constant')
            else:
                features = features[:, :max_len]

            # CNN 입력을 위한 shape: (14, 130, 1)
            features = features[..., np.newaxis]
            X.append(features)
            y.append(label)

            # 시각화
            segment_id = f"{base_filename}_seg{i+1}"
            save_path = os.path.join(output_dir, 'visuals', label_name, f"visual_{segment_id}.png")

            plt.figure(figsize=(10, 4))
            plt.imshow(features.squeeze(), aspect='auto', origin='lower', cmap='coolwarm')
            plt.title(f"{segment_id} - Label: {'quiet' if label == 0 else 'loud'}")
            plt.xlabel("Frame")
            plt.ylabel("Feature Index (MFCC + ZCR)")
            plt.colorbar(label='Feature Value')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()

# 넘파이 배열 변환 및 저장
X = np.array(X)  # shape: (샘플수, 14, 130, 1)
y = np.array(y)

np.save(os.path.join(output_dir, "X_cnn.npy"), X)
np.save(os.path.join(output_dir, "y_cnn.npy"), y)

print("✅ X shape (CNN용):", X.shape)
print("✅ y shape:", y.shape)
print(f"💾 저장 완료: outputs/cnn/X_cnn.npy, y_cnn.npy")
print("🎉 이미지도 outputs/cnn/visuals/ 폴더에 저장되었어요.")
