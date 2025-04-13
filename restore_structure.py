import os
import shutil

# 현재 기준 디렉토리
base_dir = os.getcwd()

# 되돌릴 구조 정의: {원래 위치: [현재 위치]}
reverse_structure = {
    "cnn_lstm_tf_src": ["src/tensorflow/cnn_lstm_tf_src"],
    "cnn_src": ["src/tensorflow/cnn_src"],
    "pytorch_src": ["src/pytorch/pytorch_src"],
    "Docker_tf": ["docker/tf/Docker_tf"],
    "Docker_torch": ["docker/torch/Docker_torch"],
    "models_tf": ["models/tensorflow/models_tf"],
    "cnn_lstm_pytorch.pth": ["models/pytorch/cnn_lstm_pytorch.pth"],
    "outputs_tf": ["outputs/tensorflow/outputs_tf"],
    "outputs/cnn_lstm": ["outputs/pytorch/cnn_lstm"],
    "test_audio_batch": ["test/audio_batch/test_audio_batch"],
    "test_audio_batch copy": ["test/audio_batch/test_audio_batch copy"],
    "test.py": ["test/etc/test.py"],
    "test.drawio": ["test/etc/test.drawio"],
}

# 복원 함수
def restore_items():
    print("🔁 프로젝트 되돌리기 시작!\n")
    for original, sources in reverse_structure.items():
        for src_rel in sources:
            src = os.path.join(base_dir, src_rel)
            dst = os.path.join(base_dir, original)

            if os.path.exists(src):
                shutil.move(src, dst)
                print(f"✅ Moved back: {src} → {dst}")
            else:
                print(f"⚠️ Not found: {src}")
    print("\n🎉 되돌리기 완료!")

if __name__ == "__main__":
    restore_items()
