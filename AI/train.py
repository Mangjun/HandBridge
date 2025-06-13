import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn, optim
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from datasets.dataset import SignLanguageDataset, augment_segment
from datasets.collate import collate_fn
from models.sign_model import SignModel

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 하이퍼파라미터
    BATCH_SIZE, EPOCHS, LR = 32, 5, 3e-4
    writer = SummaryWriter("logs")

    # 데이터셋
    train_ds = SignLanguageDataset(
        npz_dir="./numpy_train_data",
        label_map_path="./numpy_train_data/label_map.json",
        train=True,
        transform=augment_segment
    )
    val_ds = SignLanguageDataset(
        npz_dir="./numpy_val_data",
        label_map_path="./numpy_train_data/label_map.json",
        label_json_dir="./processed_val_data/labels",
        train=False
    )

    num_classes = len(train_ds.get_label_map())

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    # 모델 초기화
    model = SignModel(input_size=126, num_classes=num_classes, dropout=0.5).to(device)

    # 체크포인트 로드 (이미 학습된 모델이 있으면 이어서 학습)
    checkpoint_path = "checkpoints/best_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")
        saved_dict = torch.load(checkpoint_path, map_location=device)
        model_dict = model.state_dict()
        # 호환되는 파라미터만 로드
        pretrained = {k: v for k, v in saved_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
        print(f"[INFO] Loaded {len(pretrained)} parameters from checkpoint.")

    # 클래스 가중치
    counts = [0] * num_classes
    for prefix, lbl in train_ds.label_map.items():
        idx = train_ds.get_label_map()[lbl]
        counts[idx] += 1
    weights = torch.tensor([1.0 / (c + 1e-6) for c in counts], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # 옵티마이저 + 스케줄러
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=len(train_loader) * EPOCHS,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    best_val_acc, patience, counter = 0.0, 10, 0

    # 학습 루프
    for epoch in range(EPOCHS):
        print(f"\n🚀 Epoch {epoch+1}/{EPOCHS}")
        model.train()
        train_loss, train_corr, train_tot = 0.0, 0, 0
        for x, y, l in tqdm(train_loader, desc="Training"):  # batch (x, labels, lengths)
            x, y, l = x.to(device), y.to(device), l.to(device)
            optimizer.zero_grad()
            out = model(x, l)
            loss = criterion(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()

            bs = y.size(0)
            train_loss += loss.item() * bs
            preds = out.argmax(dim=1)
            train_corr += (preds == y).sum().item()
            train_tot += bs

        train_loss /= train_tot
        train_acc = train_corr / train_tot
        writer.add_scalar("Train/Loss", train_loss, epoch)
        writer.add_scalar("Train/Accuracy", train_acc, epoch)
        print(f"Train Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")

        # 검증 단계
        model.eval()
        val_loss, val_corr, val_tot = 0.0, 0, 0
        with torch.no_grad():
            for x, y, l in tqdm(val_loader, desc="Validation"):
                x, y, l = x.to(device), y.to(device), l.to(device)
                out = model(x, l)
                loss = criterion(out, y)
                bs = y.size(0)
                val_loss += loss.item() * bs
                val_corr += (out.argmax(dim=1) == y).sum().item()
                val_tot += bs

        val_loss /= val_tot
        val_acc = val_corr / val_tot
        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/Accuracy", val_acc, epoch)
        print(f"Val   Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")

        # Early Stopping & 체크포인트 저장
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✅ Saved best model (Val Acc: {val_acc:.4f})")
        else:
            counter += 1
            print(f"EarlyStopping patience {counter}/{patience}")
            if counter >= patience:
                print("⏹️ Early stopping")
                break

    writer.close()
    print("✅ Training complete")
