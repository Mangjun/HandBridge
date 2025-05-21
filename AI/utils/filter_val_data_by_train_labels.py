import os
import json
import unicodedata
from pathlib import Path
import shutil
from difflib import get_close_matches

# 학습 라벨 디렉터리
train_label_dir = Path("./processed_train_data/labels")
# 학습 키포인트 디렉터리
train_sign_data_dir = Path("./processed_train_data/sign_data")

# 검증 라벨 원본
val_label_dir = Path("./val_data/labels")
# 필터링된 검증 라벨 저장할 디렉터리
filtered_val_label_dir = Path("./filtered_val_data/labels")
# 검증 키포인트 디렉터리
val_sign_data_dir = Path("./val_data/sign_data")
# 필터링된 키포인트 저장할 디렉터리
filtered_sign_data_dir = Path("./filtered_val_data/sign_data")

# 출력 디렉터리 생성
filtered_val_label_dir.mkdir(parents=True, exist_ok=True)
filtered_sign_data_dir.mkdir(parents=True, exist_ok=True)

# 라벨 정규화 함수 정의
def normalize(label):
    return unicodedata.normalize("NFC", label.strip())

def delete_invalid_sample(file: Path, sign_data_dir: Path):
    print(f"[🗑️] 비어 있는 라벨 제거됨: {file.name}")
    file.unlink()
    base_name = file.stem.replace("_morpheme", "")
    sign_data_folder = sign_data_dir / base_name
    if sign_data_folder.exists():
        shutil.rmtree(sign_data_folder)
        print(f"[🗑️] 대응하는 sign_data 폴더 삭제됨: {sign_data_folder}")

# 학습 라벨 추출 및 유효성 검증
train_files = list(train_label_dir.glob("*_F_morpheme.json"))
total_train = len(train_files)
cnt_train = 0
train_labels = set()
print(f"[INFO] 학습 라벨 총개수: {total_train}")
for file in train_files:
    cnt_train += 1
    print(f"[TRAIN] 처리 중 {cnt_train}/{total_train}")
    try:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)

        if "data" not in data or not isinstance(data["data"], list) or len(data["data"]) == 0:
            delete_invalid_sample(file, train_sign_data_dir)
            continue

        if "attributes" not in data["data"][0] or not data["data"][0]["attributes"]:
            delete_invalid_sample(file, train_sign_data_dir)
            continue

        label = data["data"][0]["attributes"][0]["name"]
        train_labels.add(normalize(label))
    except Exception as e:
        print(f"[!] 🚨 학습 JSON 파싱 실패: {file.name} → {type(e).__name__}: {e}")
        continue

# 검증 라벨 필터링 및 복사
val_files = list(val_label_dir.glob("*_F_morpheme.json"))
total_val = len(val_files)
cnt_val = 0
kept_labels = 0
skipped_labels = 0
print(f"[INFO] 검증 라벨 총개수: {total_val}")
for file in val_files:
    cnt_val += 1
    print(f"[VAL] 처리 중 {cnt_val}/{total_val}")
    try:
        with open(file, encoding="utf-8") as f:
            data = json.load(f)

        # 유효성 검사
        if not data.get("data") or not isinstance(data["data"], list) or len(data["data"]) == 0:
            delete_invalid_sample(file, val_sign_data_dir)
            skipped_labels += 1
            continue

        if "attributes" not in data["data"][0] or not data["data"][0]["attributes"]:
            delete_invalid_sample(file, val_sign_data_dir)
            skipped_labels += 1
            continue

        label = normalize(data["data"][0]["attributes"][0]["name"])

        if label in train_labels:
            dest_label_file = filtered_val_label_dir / file.name
            if not dest_label_file.exists():
                shutil.copy(file, dest_label_file)

            base_name = file.stem.replace("_morpheme", "")
            src_sign_dir = val_sign_data_dir / base_name
            dst_sign_dir = filtered_sign_data_dir / base_name
            if src_sign_dir.exists() and not dst_sign_dir.exists():
                shutil.copytree(src_sign_dir, dst_sign_dir)

            kept_labels += 1
        else:
            print(f"[!] 매치 실패: '{label}' ← 학습 라벨에 없음")
            print(f"    ↳ 가장 유사한 라벨: {get_close_matches(label, train_labels, n=1)}")
            skipped_labels += 1
    except Exception as e:
        print(f"[!] 🚨 검증 JSON 파싱 실패: {file.name} → {type(e).__name__}: {e}")
        skipped_labels += 1
        continue

print(f"✅ 유지된 검증 샘플: {kept_labels}")
print(f"⚠️ 제외된 검증 샘플: {skipped_labels}")
print(f"📂 필터링된 검증 라벨: {filtered_val_label_dir}")
print(f"📂 필터링된 검증 데이터: {filtered_sign_data_dir}")
