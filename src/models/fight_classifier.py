"""Mô hình phân loại clip ngắn: đánh nhau / không đánh nhau.

Kiến trúc: MobileNetV3-Small trích đặc trưng từng khung hình -> ghép thêm hiệu đặc trưng
giữa hai khung liên tiếp (tín hiệu chuyển động) -> GRU hai chiều -> 1 logit.

File này được notebook `train_fight_colab.ipynb` nhúng nguyên văn, nên chỉ phụ thuộc
torch / torchvision / numpy / opencv. Sửa kiến trúc ở đây thì phải train lại.
"""
import math
from collections import deque
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

FEATURE_DIM = 576  # số kênh đầu ra của MobileNetV3-Small.features
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_CONFIG = {
    "num_frames": 16,     # số khung hình trong một clip
    "sample_fps": 8.0,    # lấy mẫu 8 khung/giây -> một clip dài 2 giây
    "img_size": 224,
    "hidden": 128,
    "dropout": 0.3,
}


class FightClipClassifier(nn.Module):
    def __init__(self, hidden: int = 128, dropout: float = 0.3, pretrained: bool = False):
        super().__init__()
        weights = "DEFAULT" if pretrained else None
        self.backbone = mobilenet_v3_small(weights=weights).features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gru = nn.GRU(FEATURE_DIM * 2, hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * 4, 1)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (N, 3, H, W) đã chuẩn hoá ImageNet -> (N, 576)."""
        return self.pool(self.backbone(frames)).flatten(1)

    def classify_features(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, T, 576) -> logit (B,). Logit > 0 nghĩa là nghiêng về 'đánh nhau'."""
        motion = feats[:, 1:] - feats[:, :-1]
        motion = torch.cat([torch.zeros_like(feats[:, :1]), motion], dim=1)
        out, _ = self.gru(torch.cat([feats, motion], dim=2))
        pooled = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=1)
        return self.head(self.dropout(pooled)).squeeze(1)

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """clips: (B, T, 3, H, W) -> logit (B,)."""
        b, t = clips.shape[:2]
        feats = self.encode_frames(clips.flatten(0, 1)).view(b, t, -1)
        return self.classify_features(feats)


def preprocess_frame(frame_bgr: np.ndarray, img_size: int) -> torch.Tensor:
    """Ảnh BGR của OpenCV -> tensor (3, S, S) đã chuẩn hoá. Co cả khung hình về hình vuông."""
    rgb = cv2.cvtColor(cv2.resize(frame_bgr, (img_size, img_size), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (x - mean) / std


class FightClipInference:
    """Chạy mô hình theo thời gian thực: mỗi ~1/sample_fps giây trích đặc trưng một khung hình,
    giữ num_frames đặc trưng gần nhất và phân loại cửa sổ đó. Chi phí mỗi lần gọi là một lượt
    MobileNetV3-Small, không phụ thuộc số người trong khung hình."""

    def __init__(self, model_path: str, device: str = "cpu"):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.config = {**DEFAULT_CONFIG, **checkpoint.get("config", {})}
        self.recommended_threshold = float(checkpoint.get("threshold", 0.5))
        self.device = torch.device(device if (device == "cpu" or torch.cuda.is_available()) else "cpu")
        self.model = FightClipClassifier(self.config["hidden"], self.config["dropout"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.interval = 1.0 / float(self.config["sample_fps"])
        self.num_frames = int(self.config["num_frames"])
        self.features = deque(maxlen=self.num_frames)  # (timestamp, tensor 576)
        self.last_probability: Optional[float] = None
        self.next_due: Optional[float] = None

    def reset(self):
        self.features.clear()
        self.last_probability = None
        self.next_due = None

    def update(self, frame_bgr: np.ndarray, timestamp: float) -> Optional[float]:
        """Đưa khung hình mới vào. Trả về xác suất đánh nhau của ~2 giây gần nhất,
        hoặc None khi chưa đủ dữ liệu liên tục."""
        if not math.isfinite(timestamp):
            return self.last_probability
        if self.features:
            gap = timestamp - self.features[-1][0]
            if gap < 0 or gap > 4 * self.interval:
                self.reset()  # thời gian lùi hoặc đứt quãng: bỏ cửa sổ cũ
            elif timestamp < self.next_due - 0.25 * self.interval:
                return self.last_probability  # chưa đến lượt lấy mẫu
        # Lấy mẫu theo lịch cố định sample_fps: nguồn 10 fps vẫn cho trung bình 8 khung/giây
        # (nếu chỉ so khoảng cách với khung trước, nguồn 10 fps bị bỏ một nửa, còn 5 khung/giây).
        if self.next_due is None or timestamp - self.next_due > self.interval:
            self.next_due = timestamp + self.interval
        else:
            self.next_due += self.interval
        x = preprocess_frame(frame_bgr, int(self.config["img_size"])).unsqueeze(0).to(self.device)
        with torch.no_grad():
            self.features.append((timestamp, self.model.encode_frames(x)[0]))
            if len(self.features) < self.num_frames:
                self.last_probability = None
            else:
                feats = torch.stack([f for _, f in self.features]).unsqueeze(0)
                self.last_probability = float(torch.sigmoid(self.model.classify_features(feats))[0])
        return self.last_probability
