import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# ✅ Optional: 한글 깨짐 방지용 (Docker에서 폰트 설치 시 사용 가능)
# import matplotlib
# matplotlib.rcParams['font.family'] = 'NanumGothic'

# 🔧 설정
X_path = "outputs/cnn_lstm/X_lstm.npy"
y_path = "outputs/cnn_lstm/y_lstm.npy"
model_save_path = "models/cnn_lstm_pytorch.pth"
plot_save_path = "outputs/cnn_lstm/train_metrics.png"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 📥 데이터 로드
X = np.load(X_path)  # (samples, 130, 14)
y = np.load(y_path)
print(f"✅ Loaded: X.shape={X.shape}, y.shape={y.shape}")
print(f"🧾 Label distribution: {np.bincount(y)}")

# 🧪 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 🧠 텐서 변환 (CNN 입력용 → (B, 1, 130, 14))
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# 🧠 모델 정의 (CNN + LSTM)
class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.lstm = nn.LSTM(input_size=64 * 3, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)  # (B, 64, 32, 3)
        x = x.permute(0, 2, 1, 3).contiguous()  # (B, 32, 64, 3)
        x = x.view(x.size(0), x.size(1), -1)    # (B, 32, 192)
        _, (h_n, _) = self.lstm(x)
        x = h_n[-1]  # 마지막 hidden state
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)

# ⚙️ 학습 설정
model = CNNLSTM().to(device)
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# 🏁 학습 루프
for epoch in range(1, 21):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor).squeeze()
    loss = loss_fn(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    preds = (outputs >= 0.5).float()
    acc = (preds == y_train_tensor).float().mean().item()

    train_losses.append(loss.item())
    train_accuracies.append(acc)

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_tensor).squeeze()
        val_loss = loss_fn(val_outputs, y_test_tensor)
        val_preds = (val_outputs >= 0.5).float()
        val_acc = (val_preds == y_test_tensor).float().mean().item()

        val_losses.append(val_loss.item())
        val_accuracies.append(val_acc)

    print(f"[{epoch:02d}] Train Loss: {loss.item():.4f} | Train Acc: {acc:.4f} | Val Loss: {val_loss.item():.4f} | Val Acc: {val_acc:.4f}")

# 💾 모델 저장
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), model_save_path)
print(f"✅ Model saved: {model_save_path}")

# 📈 시각화 (Train vs Validation 비교형 그래프)
os.makedirs("outputs/cnn_lstm", exist_ok=True)
plt.figure(figsize=(12, 6))

# Loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', marker='o')
plt.plot(val_losses, label='Validation Loss', marker='o')
plt.title("Loss (Train vs Validation)")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.legend(); plt.grid()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy', marker='x')
plt.plot(val_accuracies, label='Validation Accuracy', marker='x')
plt.title("Accuracy (Train vs Validation)")
plt.xlabel("Epoch"); plt.ylabel("Accuracy")
plt.legend(); plt.grid()

plt.tight_layout()
plt.savefig(plot_save_path)
print(f"📊 Plot saved: {plot_save_path}")
