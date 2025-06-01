import os
import json
from pathlib import Path
import shutil

# 원본 학습 데이터 경로
raw_sign_data_dir = Path("./sign_data")
raw_label_dir = Path("./labels")

# 결과 저장 경로
out_sign_data_dir = Path("./processed_train_data/sign_data")
out_label_dir = Path("./processed_train_data/labels")

# 디렉터리 생성
out_sign_data_dir.mkdir(parents=True, exist_ok=True)
out_label_dir.mkdir(parents=True, exist_ok=True)

# F 시점 라벨만 복사
count = 0
for label_file in raw_label_dir.glob("*_F_morpheme.json"):
    shutil.copy(label_file, out_label_dir / label_file.name)

    # 디렉터리 이름 구하기 (ex: NIA_SL_WORD0001_SYN01_F)
    base_name = label_file.stem.replace("_morpheme", "")
    src_sign_dir = raw_sign_data_dir / base_name
    dst_sign_dir = out_sign_data_dir / base_name

    if src_sign_dir.exists():
        shutil.copytree(src_sign_dir, dst_sign_dir, dirs_exist_ok=True)
        count += 1

print(f"✅ F 시점 학습 샘플 {count}개 복사 완료")
print(f"📂 sign_data: {out_sign_data_dir}")
print(f"📂 labels: {out_label_dir}")
