import json

with open('./numpy_train_data/label_map.json', 'r', encoding='utf-8') as f:
    prefix2label = json.load(f)

# set으로 고유 label 뽑기 → idx 부여
all_labels = sorted(set(prefix2label.values()))
label2idx = {label: idx for idx, label in enumerate(all_labels)}

# 저장
with open('./numpy_train_data/label2idx.json', 'w', encoding='utf-8') as f:
    json.dump(label2idx, f, ensure_ascii=False, indent=2)
