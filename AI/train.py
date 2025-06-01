import os
import json
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from datasets.dataset import SignLanguageDataset  # NumPy 기반 데이터셋
from datasets.collate import collate_fn
from models.sign_model import SignModel

# 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# 하이퍼파라미터
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
WARMUP_EPOCHS = 5

# TensorBoard 기록 디렉터리 설정
log_dir = "./logs"
writer = SummaryWriter(log_dir=log_dir)

# NumPy 데이터셋 로드
train_dataset = SignLanguageDataset(npz_dir="./numpy_data/train")
val_dataset   = SignLanguageDataset(npz_dir="./numpy_data/val")

# 데이터셋 체크
if len(train_dataset) == 0:
    raise RuntimeError("🚨 학습 데이터셋에 샘플이 없습니다.")
print(f"[INFO] Train samples: {len(train_dataset)}")
print(f"[INFO] Val   samples: {len(val_dataset)}")

# 클래스 수
label_map = train_dataset.get_label_map()
num_classes = len(label_map)
print(f"[INFO] Number of classes: {num_classes}")

# DataLoader 생성
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# 모델 초기화
model = SignModel(num_classes=num_classes)
model.to(device)

# 분류기 워밍업 설정: 처음 WARMUP_EPOCHS 동안 classifier만 학습
for name, param in model.named_parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if 'classifier' in name:
        param.requires_grad = True

# 초기 optimizer: classifier만
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
# 학습률 스케줄러
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# 체크포인트 로드 (optional)
checkpoint_path = "./checkpoints/best_model.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model_dict = model.state_dict()
    pretrained = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(pretrained)
    model.load_state_dict(model_dict)
    print(f"[INFO] Pretrained weights loaded: {len(pretrained)} params")

# 손실 함수
criterion = nn.CrossEntropyLoss()

# Top-k accuracy 함수
def topk_acc(output, target, k=5):
    _, pred = output.topk(k, dim=1, largest=True, sorted=True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))
    return correct.any(dim=1).float().mean().item()

best_val_accuracy = 0.0

# 학습 루프
for epoch in range(EPOCHS):
    print(f"\n🚀 Epoch {epoch+1}/{EPOCHS} 시작")

    # 워밍업 후 전체 언프리즈 및 optimizer 교체
    if epoch == WARMUP_EPOCHS:
        print("[INFO] Warm-up 완료: 전체 모델 fine-tuning 시작")
        for param in model.parameters():
            param.requires_grad = True
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE * 0.1)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels, lengths in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training"):
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, lengths)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total if total > 0 else 0.0
    writer.add_scalar("Train/Loss", train_loss, epoch)
    writer.add_scalar("Train/Accuracy", train_acc, epoch)
    print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")

    # 검증 루프
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_top5 = 0.0

    with torch.no_grad():
        for inputs, labels, lengths in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validation"):
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)
            outputs = model(inputs, lengths)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            val_top5 += topk_acc(outputs, labels, k=5) * labels.size(0)

    val_acc = val_correct / val_total if val_total > 0 else 0.0
    val_top5 = val_top5 / val_total if val_total > 0 else 0.0
    writer.add_scalar("Val/Loss", val_loss, epoch)
    writer.add_scalar("Val/Accuracy", val_acc, epoch)
    writer.add_scalar("Val/Top5", val_top5, epoch)
    print(f"[Epoch {epoch+1}] Val Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f} | Top5: {val_top5:.4f}")

    # 체크포인트 저장
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), checkpoint_path)
        print(f"✅ Best model saved (Val Acc: {val_acc:.4f})")

    # 스케줄러 스텝
    scheduler.step()

writer.close()
print("✅ 학습 완료 및 모델 저장 완료")
