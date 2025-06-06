import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # None 제거
    batch = [item for item in batch if item is not None]
    # 길이 0 제거
    batch = [item for item in batch if len(item[0]) > 0]
    if len(batch) == 0:
        return None

    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)

    lengths, sort_idx = lengths.sort(descending=True)
    sequences = [sequences[i] for i in sort_idx]
    labels = [labels[i] for i in sort_idx]

    padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)
    return padded_seqs, labels, lengths