import json
from pathlib import Path

# 라벨 추출 함수
def extract_labels_from_dir(label_dir):
    label_set = set()
    label_dir = Path(label_dir)

    for file in label_dir.glob("*_F_morpheme.json"):  # 정면(F)만
        try:
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
                label = data["data"][0]["attributes"][0]["name"]
                label_set.add(label)
        except Exception as e:
            print(f"[!] Error reading {file.name}: {e}")
    return label_set

# 학습/검증 라벨 경로
train_label_dir = Path("./processed_train_data/labels")
val_label_dir = Path("./filtered_val_data/labels")

# 라벨 추출
train_labels = extract_labels_from_dir(train_label_dir)
val_labels = extract_labels_from_dir(val_label_dir)

# 비교
only_in_train = train_labels - val_labels
only_in_val = val_labels - train_labels

# 출력
print("✅ 학습 라벨 수:", len(train_labels))
print("✅ 검증 라벨 수:", len(val_labels))

if not only_in_train and not only_in_val:
    print("🎉 학습 라벨과 검증 라벨이 완전히 일치합니다.")
else:
    if only_in_train:
        print("\n⚠️ 학습셋에만 있는 라벨:")
        for label in sorted(only_in_train):
            print(" -", label)

    if only_in_val:
        print("\n⚠️ 검증셋에만 있는 라벨:")
        for label in sorted(only_in_val):
            print(" -", label)
