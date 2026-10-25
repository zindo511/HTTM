"""
LUỒNG HUẤN LUYỆN MẠNG BI-LSTM (OFFLINE TRAINING SCRIPT)
------------------------------------------------------
Chạy trên Google Colab hoặc máy tính có GPU:
1. Đọc dữ liệu X_dataset.npy và y_dataset.npy.
2. Huấn luyện 50 Epochs với Optimizer Adam, Loss Categorical Cross-Entropy.
3. Xuất file trọng số weights/fall_bilstm.pt.
4. Đánh giá và lưu biểu đồ Loss/Accuracy cùng Ma trận nhầm lẫn (Confusion Matrix 3x3).
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np

# Thử import matplotlib / seaborn để vẽ đồ thị nếu có
try:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report
    import seaborn as sns
    HAS_PLOT_LIB = True
except ImportError:
    HAS_PLOT_LIB = False

from src.models.bilstm_verifier import BiLSTMActionClassifier


def train_model(
    data_x_path: str = "data/X_dataset.npy",
    data_y_path: str = "data/y_dataset.npy",
    weights_output_path: str = "weights/fall_bilstm.pt",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Sử dụng thiết bị huấn luyện: {device}")
    
    if not os.path.exists(data_x_path) or not os.path.exists(data_y_path):
        print(f"[WARNING] Không tìm thấy dữ liệu tại {data_x_path}. Tạo dữ liệu giả lập (Dummy Data) để kiểm thử luồng code...")
        # Tạo 300 mẫu giả lập: 300 mẫu, 30 frames, 34 đặc trưng, 3 nhãn
        X = np.random.randn(300, 30, 34).astype(np.float32)
        y = np.random.randint(0, 3, size=(300,)).astype(np.int64)
    else:
        X = np.load(data_x_path)
        y = np.load(data_y_path)
        
    print(f"[INFO] Tổng số mẫu: {len(X)} | Kích thước dữ liệu: {X.shape}")
    
    # Chia tập Train (70%), Val (15%), Test (15%)
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    
    train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    # Khởi tạo mô hình Bi-LSTM
    model = BiLSTMActionClassifier(input_dim=34, hidden_dim=64, num_classes=3, dropout_rate=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    
    print("\n--- BẮT ĐẦU HUẤN LUYỆN 50 EPOCHS ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)
            
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
        # Đánh giá trên tập Validation
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_running_loss += loss.item() * batch_x.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)
                
        val_loss = val_running_loss / total_val
        val_acc = correct_val / total_val
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs}] - Train Loss: {train_loss:.4f}, Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f}, Acc: {val_acc*100:.1f}%")
            
        # Lưu checkpoint mô hình tốt nhất
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(weights_output_path), exist_ok=True)
            torch.save(model.state_dict(), weights_output_path)
            
    print(f"\n[SUCCESS] Huấn luyện hoàn tất! Trọng số tối ưu đã được lưu tại: {weights_output_path}")
    
    # Đánh giá trên tập Test
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
            
    if HAS_PLOT_LIB and len(all_targets) > 0:
        target_names = ["Normal (0)", "Running (1)", "Falling (2)"]
        print("\n--- BÁO CÁO PHÂN LOẠI TRÊN TẬP TEST ---")
        print(classification_report(all_targets, all_preds, target_names=target_names))
        
        # Vẽ Ma trận nhầm lẫn
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
        plt.title("Confusion Matrix 3x3 - Test Set")
        plt.xlabel("Dự đoán (Predicted)")
        plt.ylabel("Thực tế (Ground Truth)")
        plt.tight_layout()
        os.makedirs("docs", exist_ok=True)
        plt.savefig("docs/confusion_matrix.png")
        print("[INFO] Đã lưu ảnh ma trận nhầm lẫn vào docs/confusion_matrix.png")


if __name__ == "__main__":
    train_model(epochs=50)
