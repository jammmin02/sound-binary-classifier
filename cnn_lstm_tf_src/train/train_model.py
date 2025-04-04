import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Reshape, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import os

# 🔧 Config
X_path = "outputs/cnn_lstm/X_lstm.npy"
y_path = "outputs/cnn_lstm/y_lstm.npy"
model_save_path = "cnn_lstm_model.h5"
plot_save_path = "cnn_lstm_train_history.png"

# 📥 Load data
X = np.load(X_path)  # (샘플, 130, 14)
y = np.load(y_path)
print(f"✅ Data loaded: X shape = {X.shape}, y shape = {y.shape}")
print(f"🧾 Label distribution: {np.bincount(y)}")  # class imbalance 확인

# 📐 Reshape for CNN2D
X = X[..., np.newaxis]  # (샘플, 130, 14, 1)

# 📊 Split dataset
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2/9, stratify=y_temp, random_state=42)

print(f"📊 Split sizes → Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# 🧠 Build model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(130, 14, 1)),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2)),

    # Reshape to LSTM input
    Reshape((32, -1)),  # ex) (batch, 32, 112)

    LSTM(64),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 🏁 Train model
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# → 'val_accuracy'로 바꿔도 OK

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# 🧪 Evaluate model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"🧪 Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

# 💾 Save model
model.save(model_save_path)
print(f"✅ Model saved: {model_save_path}")

# 📊 Plot training history
plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='x')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='x')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig(plot_save_path)
plt.show()
print(f"📈 Plot saved: {plot_save_path}")
