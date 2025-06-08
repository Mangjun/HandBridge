import numpy as np
import sys
from pathlib import Path

def load_npz(file_path: str) -> None:
    """NPZ 파일의 내용을 확인합니다."""
    try:
        # NPZ 파일 로드
        data = np.load(file_path, allow_pickle=True)
        
        print("\n=== NPZ 파일 내용 ===")
        print(f"파일 경로: {file_path}")
        
        # 저장된 배열 목록
        print("\n포함된 배열:")
        for key in data.files:
            array = data[key]
            print(f"\n[{key}]")
            print(f"Shape: {array.shape}")
            print(f"Type: {array.dtype}")
            
            # 키포인트 데이터 상세 정보
            if key == "keypoints":
                print("\n키포인트 정보:")
                print(f"프레임 수: {len(array)}")
                print(f"프레임당 특징 수: {array.shape[1]}")  # 126 = 21개 랜드마크 × 3좌표 × 2손
                
                # 첫 번째 프레임의 데이터 예시
                print("\n첫 번째 프레임 샘플 (처음 10개 값):")
                print(array[0][:10])
                
                # 영점이 아닌 프레임 수 계산 (손이 감지된 프레임)
                non_zero_frames = np.count_nonzero(np.any(array != 0, axis=1))
                print(f"\n손이 감지된 프레임 수: {non_zero_frames}/{len(array)}")
            
            # 비디오 정보 출력
            elif key == "video_info":
                print("\n비디오 정보:")
                for item in array:
                    print(f"{item[0]}: {item[1]}")

    except Exception as e:
        print(f"에러 발생: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python check_npz.py <npz_파일_경로>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    if not Path(file_path).exists():
        print(f"파일을 찾을 수 없습니다: {file_path}")
        sys.exit(1)
    
    load_npz(file_path) 