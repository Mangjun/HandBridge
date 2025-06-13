import os
import glob
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import random
import cv2
import mediapipe as mp

# 1) Transformer 기반 수어 분류 모델 정의
class SignTransformer(nn.Module):
    def __init__(self,
                 input_dim=126,
                 d_model=256,
                 nhead=8,
                 num_layers=4,
                 num_classes=3000,
                 max_len=300,
                 dropout=0.1):
        super(SignTransformer, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):  # x: [B, T, F]
        B, T, _ = x.size()
        # 포지션 임베딩 슬라이스
        x = self.input_proj(x) + self.pos_embed[:, :T, :]
        y = self.transformer(x)
        y = y.mean(dim=1)
        return self.head(y)

# 2) KeypointDataset: npz → 시퀀스 + 라벨 처리 + augmentation 
class KeypointDataset(Dataset):
    def __init__(self,
                 npz_dir,
                 label_map_path,
                 multilabel=False,
                 sentence=False,
                 max_len=300,
                 window_size=30,
                 train=False):
        super(KeypointDataset, self).__init__()
        self.multilabel = multilabel
        self.sentence = sentence
        self.train = train
        with open(label_map_path, 'r', encoding='utf-8') as f:
            self.prefix2label = json.load(f)
        labels = sorted(set(self.prefix2label.values()))
        self.idx2label = labels
        self.label2idx = {v: i for i, v in enumerate(labels)}
        self.items = []
        for fname in sorted(os.listdir(npz_dir)):
            if not fname.endswith('.npz'): continue
            prefix = os.path.splitext(fname)[0]
            npz_path = os.path.join(npz_dir, fname)
            if not sentence:
                if prefix not in self.prefix2label: continue
                lbl = self.prefix2label[prefix]
                if lbl not in self.label2idx: continue
                self.items.append((npz_path, [lbl]))
            else:
                jp = os.path.join(
                    os.path.dirname(label_map_path).replace('numpy_train_data','video_path/labels'),
                    f"{prefix}_morpheme.json"
                )
                if not os.path.exists(jp): continue
                segs = json.load(open(jp, 'r', encoding='utf-8'))['data']
                seg_labels = [seg['attributes'][0]['name'] for seg in segs]
                if not seg_labels: continue
                self.items.append((npz_path, seg_labels))
        self.max_len = max_len
        self.window_size = window_size

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        npz_path, seg_labels = self.items[idx]
        arr = np.load(npz_path)['keypoints']  # [T,126]
        # 길이 자르거나 패딩
        T, F = arr.shape
        if T > self.max_len:
            idxs = np.linspace(0, T-1, num=self.max_len, dtype=int)
            arr = arr[idxs]
        else:
            pad = np.zeros((self.max_len-T, F), dtype=arr.dtype)
            arr = np.vstack([arr, pad])
        # augmentation: 가우시안 노이즈
        if self.train:
            if random.random() < 0.5:
                arr += np.random.normal(0, 0.01, arr.shape)
        x = torch.from_numpy(arr).float()
        # 라벨
        if self.multilabel:
            y = torch.zeros(len(self.idx2label), dtype=torch.float)
            for w in seg_labels:
                if w in self.label2idx:
                    y[self.label2idx[w]] = 1.0
        else:
            y = torch.tensor(self.label2idx[seg_labels[0]], dtype=torch.long)
        return x, y

# 3) Custom collate_fn: invalid shape 필터링

def collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch: return None
    xs, ys = zip(*batch)
    x = torch.stack(xs, dim=0)
    y = torch.stack(ys, dim=0)
    return x, y

# 4) 학습 / 파인튜닝 루틴 with label smoothing & early stopping

def run_training(
    npz_dir,
    label_map,
    ckpt_in,
    ckpt_out,
    multilabel=False,
    sentence=False,
    epochs=10,
    batch_size=32,
    lr=3e-4,
    weight_decay=1e-5,
    smoothing=0.1,
    patience=3
):
    ds = KeypointDataset(
        npz_dir,
        label_map,
        multilabel,
        sentence,
        max_len=300,
        window_size=30,
        train=True
    )
    ds_val = KeypointDataset(
        npz_dir.replace('train','sentence' if sentence else 'train'),
        label_map,
        multilabel,
        sentence,
        max_len=300,
        window_size=30,
        train=False
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=not multilabel,
                        num_workers=4, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignTransformer(
        input_dim=126,
        d_model=256,
        nhead=8,
        num_layers=2,
        num_classes=len(ds.idx2label),
        max_len=300,
        dropout=0.1
    ).to(device)
    # ckpt 로드
    if ckpt_in and os.path.exists(ckpt_in):
        state = torch.load(ckpt_in, map_location=device)
        md = model.state_dict()
        for k,v in state.items():
            if k in md and v.size()==md[k].size(): md[k]=v
        model.load_state_dict(md)
        print(f"Loaded checkpoint: {ckpt_in}")

    # 손실
    if multilabel:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=smoothing)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=len(loader)*epochs,
        pct_start=0.3
    )

    best_val_loss = float('inf')
    counter = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
            if batch is None: continue
            x,y = batch
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * x.size(0)
        train_loss = total_loss / len(ds)

        # 검증
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                if batch is None: continue
                x,y = batch
                x,y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                if not multilabel:
                    preds = logits.argmax(dim=1)
                    correct += (preds==y).sum().item()
                    total += y.size(0)
        val_loss /= len(ds_val)
        val_acc = correct/total if total>0 else 0
        print(
            f"Epoch {epoch+1}: TrainLoss={train_loss:.4f}, "
            f"ValLoss={val_loss:.4f}, ValAcc={val_acc:.4f}"
        )

        # EarlyStopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), ckpt_out)
            print(f"Saved best model: {ckpt_out}")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

# 5) 실행 예시
if __name__ == '__main__':
    run_training(
        npz_dir='./numpy_train_data',
        label_map='./numpy_train_data/label_map.json',
        ckpt_in='',
        ckpt_out='checkpoints/transformer_pretrained.pth',
        multilabel=False,
        sentence=False,
        epochs=5,
        batch_size=32,
        smoothing=0.1,
        patience=2
    )
    run_training(
        npz_dir='./numpy_sentence_data',
        label_map='./numpy_train_data/label_map.json',
        ckpt_in='checkpoints/transformer_pretrained.pth',
        ckpt_out='checkpoints/transformer_finetuned.pth',
        multilabel=True,
        sentence=True,
        epochs=5,
        batch_size=32,
        smoothing=0.1,
        patience=2
    )
