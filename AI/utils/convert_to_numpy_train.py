import os
import json
import numpy as np
import unicodedata
from pathlib import Path

# ==== 설정 부분만 필요에 맞게 바꾸세요 ====
SIGN_DIR = Path("./processed_train_data/sign_data")
LABEL_DIR = Path("./processed_train_data/labels")
OUTPUT_DIR = Path("./numpy_data/train")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LABEL_MAP_PATH = OUTPUT_DIR / "label_map.json"
# ========================================

# 유니코드 정규화 함수
def normalize(label):
    return unicodedata.normalize("NFC", label.strip())

label_map = {}

# 라벨 파일 목록 및 총 개수
label_files = list(LABEL_DIR.glob("*_F_morpheme.json"))
total_labels = len(label_files)
print(f"📂 총 라벨 파일 수: {total_labels}")

# 1) 라벨맵 및 .npz 파일 생성
for idx, label_file in enumerate(label_files, 1):
    print(f"🔄 처리 중 {idx}/{total_labels}: {label_file.name}")
    with open(label_file, encoding="utf-8-sig") as lf:
        meta = json.load(lf)

    # 유효성 검사
    if not meta.get("data") or not meta["data"][0].get("attributes"):
        print(f"   ⚠️ 유효하지 않은 메타데이터, 스킵: {label_file.name}")
        continue

    label_name = normalize(meta["data"][0]["attributes"][0]["name"])
    if label_name not in label_map:
        label_map[label_name] = len(label_map)
    label_idx = label_map[label_name]

    base = label_file.stem.replace("_morpheme", "")
    folder = SIGN_DIR / base
    if not folder.exists():
        print(f"   ⚠️ 키포인트 폴더 없음, 스킵: {folder}")
        continue

    frames = []
    json_files = sorted(folder.glob("*_F_*.json"))
    total_frames = len(json_files)
    print(f"   ▶ 총 프레임 수: {total_frames}")

    # 프레임별 처리
    for f_idx, jf in enumerate(json_files, 1):
        # 진행률 표시
        if f_idx % 10 == 0 or f_idx == total_frames:
            print(f"      ⏳ 프레임 처리 {f_idx}/{total_frames}")
        try:
            with open(jf, encoding="utf-8-sig") as f:
                data = json.load(f)
        except Exception:
            continue
        people = data.get("people", {})
        left = people.get("hand_left_keypoints_2d", [])
        right = people.get("hand_right_keypoints_2d", [])
        if left and right:
            frames.append(left + right)

    if len(frames) < 5:
        print(f"   ⚠️ 시퀀스 길이 부족 ({len(frames)}), 스킵: {base}")
        continue

    arr = np.array(frames, dtype=np.float32)
    out_path = OUTPUT_DIR / f"{base}.npz"
    np.savez_compressed(out_path, keypoints=arr, label=label_idx)

# 2) 라벨맵 JSON으로 저장
with open(LABEL_MAP_PATH, 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print(f"✅ 변환 완료: {len(label_map)} classes, {len(list(OUTPUT_DIR.glob('*.npz')))} files")
print(f"✅ 라벨맵 저장: {LABEL_MAP_PATH}")
