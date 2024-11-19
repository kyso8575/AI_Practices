import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Sine 파형 데이터 생성
def create_sine_wave_data(seq_length, num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        # 임의의 시작점 설정
        start = np.random.rand()
        # 시퀀스 설정
        x = np.linspace(start, start + 2 * np.pi, seq_length)
        # 사인파 계산
        X.append(np.sin(x))
        y.append(np.sin(x + 0.1))
    return np.array(X), np.array(y)

seq_length = 50
num_samples = 1000
X, y = create_sine_wave_data(seq_length, num_samples)


# 데이터셋을 PyTorch 텐서로 변환, 원래는 (num_samples, seq_length) 크기의 텐서를 가짐, 얘를 (1000, 50, 1)의 텐서로 변환
X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)

# 타겟 텐서를 마지막 차원 제거하여 (batch_size, 1)로 변경
y = y[:, -1, :]


class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size)  # 초기 은닉 상태
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])  # 마지막 시간 단계의 출력
        return out

input_size = 1
hidden_size = 32
output_size = 1
model = SimpleRNN(input_size, hidden_size, output_size)


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), hidden_size)  # 초기 은닉 상태
        c0 = torch.zeros(1, x.size(0), hidden_size)  # 초기 셀 상태
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 마지막 시간 단계의 출력
        return out

model1 = SimpleLSTM(input_size, hidden_size, output_size)
model2 = SimpleRNN(input_size, hidden_size, output_size)

criterion = nn.MSELoss()
optimizer1 = optim.Adam(model1.parameters(), lr=0.01)
optimizer2 = optim.Adam(model2.parameters(), lr=0.01)

# 모델 학습
num_epochs = 100
losses1 = []
losses2 = []

for epoch in range(num_epochs):
    # 모델1 학습
    model1.train()
    outputs1 = model1(X)
    optimizer1.zero_grad()
    loss1 = criterion(outputs1, y)
    loss1.backward()
    optimizer1.step()
    losses1.append(loss1.item())

    # 모델2 학습
    model2.train()
    outputs2 = model2(X)
    optimizer2.zero_grad()
    loss2 = criterion(outputs2, y)
    loss2.backward()
    optimizer2.step()
    losses2.append(loss2.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Model1 Loss: {loss1.item():.4f}, Model2 Loss: {loss2.item():.4f}')

print('Finished Training')

# 모델 평가
model1.eval()
model2.eval()
with torch.no_grad():
    predicted1 = model1(X).detach().numpy()
    predicted2 = model2(X).detach().numpy()

# 시각화
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(y.numpy().flatten(), label='True')
plt.plot(predicted1.flatten(), label='Predicted by LSTM')
plt.title('LSTM Predictions')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y.numpy().flatten(), label='True')
plt.plot(predicted2.flatten(), label='Predicted by RNN')
plt.title('RNN Predictions')
plt.legend()

plt.show()

# Loss 시각화
plt.figure(figsize=(10, 5))
plt.plot(losses1, label='LSTM Loss')
plt.plot(losses2, label='RNN Loss')
plt.title('Training Loss')
plt.legend()
plt.show()