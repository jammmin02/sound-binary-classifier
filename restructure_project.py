import os
import shutil

# 기존 디렉토리 기준
base_dir = os.getcwd()

# 이동할 폴더 구조 정의
new_structure = {
    "src/tensorflow": ["cnn_lstm_tf_src", "cnn_src"],
    "src/pytorch": ["pytorch_src"],
    "docker/tf": ["Docker_tf"],
    "docker/torch": ["Docker_torch"],
    "models/tensorflow": ["models_tf"],
    "models/pytorch": ["cnn_lstm_pytorch.pth"],  # ✅ 여기 수정!
    "outputs/tensorflow": ["outputs_tf"],
    "outputs/pytorch": ["outputs/cnn_lstm"],
    "test/audio_batch": ["test_audio_batch", "test_audio_batch copy"],
    "test/etc": ["test.py", "test.drawio"],
}


# 폴더 생성 + 이동 함수
def move_items(target_folder, items):
    abs_target = os.path.join(base_dir, target_folder)
    os.makedirs(abs_target, exist_ok=True)

    for item in items:
        src = os.path.join(base_dir, item)
        dst = os.path.join(abs_target, os.path.basename(item))

        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"✅ Moved: {src} → {dst}")
        else:
            print(f"⚠️ Not found: {src}")

# 실행
if __name__ == "__main__":
    print("📁 프로젝트 폴더 재배치 시작!\n")
    for new_folder, items in new_structure.items():
        move_items(new_folder, items)
    print("\n🎉 재배치 완료!")
