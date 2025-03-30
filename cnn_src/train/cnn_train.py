import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo-', label='Train Acc')
    plt.plot(epochs, val_acc, 'ro-', label='Val Acc')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo-', label='Train Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("cnn_train_history.png")
    plt.show()
    print("📸 그래프 저장 완료: cnn_train_history.png")


# 1. 데이터 로딩
X = np.load("outputs/cnn/X_cnn.npy")
y = np.load("outputs/cnn/y_cnn.npy")

# 2. 데이터 3-way 분할 (Train 70% / Val 20% / Test 10%)

# 먼저 test 10% 분리
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.1, stratify=y, random_state=42
)

# 남은 90% 중에서 train 70%, val 20% → val은 2/9 (약 22.2%)로 분리
val_ratio = 2 / 9
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=42
)

print(f"📊 데이터 분할 결과:")
print(f"Train: {X_train.shape[0]} / Val: {X_val.shape[0]} / Test: {X_test.shape[0]}")

# 3. CNN 모델 구성
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(14, 130, 1), padding='same'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 4. 학습
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop]
)

# 5. 테스트셋 평가
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"🧪 최종 테스트 정확도: {test_acc:.4f} | 손실: {test_loss:.4f}")

# 6. 모델 저장
model.save("cnn_model.h5")
# 실행
plot_history(history)
print("✅ CNN 모델 저장 완료: cnn_model.h5")

