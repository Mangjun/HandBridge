from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

# 원본 학습 데이터 경로
raw_sign_data_dir = Path("./val_sign_data")
raw_label_dir = Path("./val_labels")

# 결과 저장 경로
out_sign_data_dir = Path("./processed_val_data/sign_data")
out_label_dir = Path("./processed_val_data/labels")

# 디렉터리 생성
out_sign_data_dir.mkdir(parents=True, exist_ok=True)
out_label_dir.mkdir(parents=True, exist_ok=True)

def copy_sample(label_file):
    try:
        # 라벨 파일 복사 (이미 있으면 스킵)
        dst_label_file = out_label_dir / label_file.name
        if not dst_label_file.exists():
            shutil.copy(label_file, dst_label_file)
        else:
            return 0  # 이미 있으면 스킵

        # 영상 파일 이름 구하기 (ex: NIA_SL_WORD0001_SYN01_F.mp4)
        base_name = label_file.stem.replace("_morpheme", ".mp4")
        src_sign_file = raw_sign_data_dir / base_name
        dst_sign_file = out_sign_data_dir / base_name

        # 영상 파일 복사 (이미 있으면 스킵)
        if src_sign_file.exists() and src_sign_file.is_file():
            if not dst_sign_file.exists():
                shutil.copy(src_sign_file, dst_sign_file)
                return 1  # 성공적으로 복사한 경우 카운트
            else:
                return 0  # 이미 있으면 스킵
        return 0
    except Exception as e:
        print(f"❌ 오류: {label_file}: {e}")
        return 0

label_files = list(raw_label_dir.glob("*_F_morpheme.json"))
count = 0

with ThreadPoolExecutor(max_workers=8) as executor:
    futures = [executor.submit(copy_sample, label_file) for label_file in label_files]
    for f in as_completed(futures):
        count += f.result()

print(f"✅ F 시점 검증 샘플 {count}개 복사 완료")
print(f"📂 sign_data: {out_sign_data_dir}")
print(f"📂 labels: {out_label_dir}")
