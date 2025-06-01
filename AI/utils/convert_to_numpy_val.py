import json
import numpy as np
import unicodedata
from pathlib import Path

# ==== 설정 부분만 필요에 맞게 바꾸세요 ====  
SIGN_DIR = Path("./filtered_val_data/sign_data")  # 검증 keypoint 디렉터리  
LABEL_DIR = Path("./filtered_val_data/labels")    # 검증 라벨 디렉터리  
OUTPUT_DIR = Path("./numpy_data/val")    # 출력 디렉터리  
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LABEL_MAP_PATH = OUTPUT_DIR / "label_map.json"
# ========================================

# 유니코드 정규화 함수
def normalize(label):
    return unicodedata.normalize("NFC", label.strip())

label_map = {}

# 1) 검증 라벨 목록 및 총 개수
label_files = list(LABEL_DIR.glob("*_F_morpheme.json"))
total = len(label_files)
print(f"📂 검증 라벨 총개수: {total}")

# 2) 각 라벨 파일 처리
for idx, label_file in enumerate(label_files, 1):
    print(f"🔄 처리 중 {idx}/{total}: {label_file.name}")
    with open(label_file, encoding="utf-8-sig") as lf:
        meta = json.load(lf)

    # 메타 유효성 검사
    if not meta.get("data") or not meta["data"][0].get("attributes"):
        print(f"   ⚠️ 유효하지 않은 메타데이터, 스킵: {label_file.name}")
        continue

    # 라벨명 정규화 및 맵핑
    label_name = normalize(meta["data"][0]["attributes"][0]["name"])
    if label_name not in label_map:
        label_map[label_name] = len(label_map)
    label_idx = label_map[label_name]

    # keypoint 폴더
    base = label_file.stem.replace("_morpheme", "")
    folder = SIGN_DIR / base
    if not folder.exists():
        print(f"   ⚠️ 키포인트 폴더 없음, 스킵: {folder}")
        continue

    frames = []
    jsons = sorted(folder.glob("*_F_*.json"))
    total_frames = len(jsons)
    print(f"   ▶ 총 프레임 수: {total_frames}")

    # 3) 프레임별 처리
    for f_idx, jf in enumerate(jsons, 1):
        # 진행률 표시
        if f_idx % 10 == 0 or f_idx == total_frames:
            print(f"      ⏳ 프레임 {f_idx}/{total_frames} 처리")
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

    # 최소 프레임 수 확인
    if len(frames) < 5:
        print(f"   ⚠️ 시퀀스 길이 부족 ({len(frames)}), 스킵: {base}")
        continue

    # NumPy 배열 저장
    arr = np.array(frames, dtype=np.float32)
    out_path = OUTPUT_DIR / f"{base}.npz"
    np.savez_compressed(out_path, keypoints=arr, label=label_idx)

# 4) 라벨맵 JSON 저장
with open(LABEL_MAP_PATH, 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=2)

print(f"✅ 검증 데이터 변환 완료: {len(label_map)} classes, {len(list(OUTPUT_DIR.glob('*.npz')))} files")
print(f"✅ 검증 라벨맵 저장: {LABEL_MAP_PATH}")
