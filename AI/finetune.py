import os
import json
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from datasets.sentence_dataset import SentenceSegmentDataset
from datasets.dataset import augment_segment
from models.sign_model import SignModel

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 경로 설정 (npz 사용)
    NPZ_DIR   = "./numpy_sentence_data"
    LABEL_DIR = "./video_path/labels"
    PRETRAIN_CKPT = "checkpoints/best_model.pth"
    FINETUNE_CKPT = "checkpoints/fine_tuned.pth"

    # 라벨맵 로드
    with open("./numpy_train_data/label_map.json", 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    word2idx = {v: i for i, v in enumerate(sorted(set(label_map.values())))}

    # 데이터셋 준비
    train_ds = SentenceSegmentDataset(
        npz_dir=NPZ_DIR,
        label_dir=LABEL_DIR,
        label_map=word2idx,
        transform=augment_segment,
        window_size=30,
        target_fps=30
    )
    if len(train_ds) == 0:
        raise RuntimeError("SentenceSegmentDataset에 데이터가 없습니다.")

    # 클래스 불균형 대응 샘플러
    counts = [0] * len(word2idx)
    for it in train_ds.items:
        counts[it['label']] += 1
    sample_weights = [1.0 / (counts[it['label']] + 1e-6) for it in train_ds.items]
    sampler = WeightedRandomSampler(sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)

    loader = DataLoader(
        train_ds,
        batch_size=32,
        sampler=sampler,
        num_workers=0,
        pin_memory=True
    )

    # 모델 선언
    model = SignModel(input_size=126, num_classes=len(word2idx), dropout=0.5).to(device)

    # 1) 사전 학습된 best_model 불러오기 (있으면)
    if os.path.exists(PRETRAIN_CKPT):
        print(f"[INFO] Loading pre-trained checkpoint from {PRETRAIN_CKPT}")
        state = torch.load(PRETRAIN_CKPT, map_location=device)
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in state.items()
                      if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
        print(f"[INFO] Loaded {len(pretrained)} params from pre-trained checkpoint.")

    # 모든 파라미터 학습 모드로 설정
    for param in model.parameters():
        param.requires_grad = True

    # 손실 및 옵티마이저/스케줄러 설정
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-4,
        total_steps=len(loader) * 20,
        pct_start=0.1
    )

    # 2) 기존 파인튜닝 체크포인트 불러오기 (있으면)
    if os.path.exists(FINETUNE_CKPT):
        print(f"[INFO] Loading fine-tuned checkpoint from {FINETUNE_CKPT}")
        state = torch.load(FINETUNE_CKPT, map_location=device)
        model_dict = model.state_dict()
        pretrained = {k: v for k, v in state.items()
                      if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained)
        model.load_state_dict(model_dict)
        print(f"[INFO] Loaded {len(pretrained)} params from fine-tune checkpoint.")

    # Fine-tuning loop (20 epochs)
    for epoch in range(20):
        model.train()
        train_loss, train_corr, train_tot = 0.0, 0, 0
        desc = f"Fine-tune Epoch {epoch+1}/20"
        for x, y in tqdm(loader, desc=desc):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            lengths = torch.full((x.size(0),), train_ds.target_fps,
                                 dtype=torch.long).to(device)
            out = model(x, lengths)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            bs = y.size(0)
            train_loss += loss.item() * bs
            preds = out.argmax(dim=1)
            train_corr += (preds == y).sum().item()
            train_tot += bs

        train_loss /= train_tot
        train_acc = train_corr / train_tot
        print(f"[{epoch+1}] Fine-tune Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")

    # 체크포인트 저장
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), FINETUNE_CKPT)
    print("✅ Fine-tuning 완료, 모델 저장됨:", FINETUNE_CKPT)
