import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class BiLSTMActionClassifier(nn.Module):
    """
    Mạng Bi-LSTM phân loại hành động dựa trên chuỗi 17 khớp xương (Tier 2 Verifier).
    Đầu vào: (Batch_size, 30, 34) - Cửa sổ trượt 30 frames của 17 khớp chuẩn hóa.
    Đầu ra: (Batch_size, 3) - Logits 3 lớp: [Normal, Running, Falling].
    """
    def __init__(self, input_dim: int = 34, hidden_dim: int = 64, num_classes: int = 3, dropout_rate: float = 0.3):
        super(BiLSTMActionClassifier, self).__init__()
        
        # Lớp Bi-LSTM hai chiều (Forward + Backward)
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout_rate)
        # 64 units * 2 chiều = 128 chiều đầu ra
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 30, 34)
        lstm_out, (hn, cn) = self.bilstm(x)
        # Lấy hidden state ở bước thời gian cuối cùng
        last_out = lstm_out[:, -1, :]  # shape: (batch_size, 128)
        out = self.dropout(last_out)
        logits = self.fc(out)
        return logits


class BiLSTMInferenceWrapper:
    """
    Bộ bọc suy luận (Inference Wrapper) phục vụ chạy thời gian thực trên CPU hoặc GPU.
    """
    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = BiLSTMActionClassifier().to(self.device)
        self.model.eval()
        
        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"[INFO] Tải trọng số Bi-LSTM thành công từ: {model_path}")
            except Exception as e:
                print(f"[WARNING] Không thể tải model Bi-LSTM từ {model_path} ({e}). Sẽ sử dụng cấu hình mặc định.")

    def predict(self, sequence_30x34: np.ndarray) -> Tuple[float, float, float]:
        """
        Dự đoán xác suất 3 lớp: Normal, Running, Falling.
        Đầu vào: numpy array shape (1, 30, 34) hoặc (30, 34)
        """
        if sequence_30x34.ndim == 2:
            sequence_30x34 = np.expand_dims(sequence_30x34, axis=0)
            
        tensor_x = torch.from_numpy(sequence_30x34).float().to(self.device)
        with torch.no_grad():
            probs = torch.softmax(self.model(tensor_x), dim=1).cpu().numpy()[0]
            
        # probs: [p_normal, p_running, p_falling]
        return float(probs[0]), float(probs[1]), float(probs[2])
