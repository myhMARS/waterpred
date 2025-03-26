import torch
import torch.nn as nn
import joblib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Waterlevel_Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(Waterlevel_Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1),
        )

        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out[:, -1, :]


class ModelMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

        # 1. 加载 PyTorch 模型（仅在 Django 启动时执行一次）
        self.model = Waterlevel_Model(8, 64, 6).to(device)
        self.model.load_state_dict(torch.load('./api/model/waterlevel_model_8_64_6.pt'))  # 替换为你的模型路径
        self.model.eval()  # 设为评估模式，不会影响权重
        self.scaler = joblib.load('./api/model/scaler.pkl')
        for _ in self.scaler:
            _.feature_names_in_ = None

    def __call__(self, request):
        # 2. 将模型绑定到 request 对象，使其在整个请求过程中可用
        request.model = self.model
        request.scaler = self.scaler
        request.device = device

        # 处理请求
        response = self.get_response(request)
        return response
