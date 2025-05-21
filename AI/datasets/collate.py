import torch
from torch.nn.utils.rnn import pad_sequence

# 데이터로더에서 사용하는 collate 함수 정의
def collate_fn(batch):
    """
    배치 데이터를 받아 padding하고 텐서로 묶는 함수
    batch: list of (keypoints_tensor [seq_len, 126], label_tensor)

    return:
        padded_seqs: [batch_size, max_seq_len, 126]
        labels: [batch_size]
        lengths: [batch_size]  # 원래 각 시퀀스 길이 (RNN에 유용)
    """
    # 배치에서 keypoints 시퀀스와 라벨 분리
    sequences = [item[0] for item in batch]  # [seq_len, 126]
    labels = [item[1] for item in batch]     # 정수 라벨

    # 시퀀스 길이 기록 (RNN 입력 시 pack_padded_sequence에 사용 가능)
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

    # 가장 긴 시퀀스 기준으로 padding → [batch_size, max_seq_len, 126]
    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)

    # 라벨 리스트를 텐서로 변환 → [batch_size]
    labels = torch.stack(labels)

    return padded_seqs, labels, lengths
