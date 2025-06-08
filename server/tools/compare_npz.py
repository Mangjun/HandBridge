import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt

def compare_npz_files(file1: str, file2: str) -> None:
    """두 NPZ 파일의 키포인트를 비교합니다."""
    try:
        # NPZ 파일 로드
        data1 = np.load(file1, allow_pickle=True)
        data2 = np.load(file2, allow_pickle=True)
        
        # 키포인트 데이터 추출
        keypoints1 = data1['keypoints']
        keypoints2 = data2['keypoints']
        
        print("\n=== 기본 정보 비교 ===")
        print(f"파일1: {Path(file1).name}")
        print(f"- Shape: {keypoints1.shape}")
        print(f"\n파일2: {Path(file2).name}")
        print(f"- Shape: {keypoints2.shape}")
        
        # 프레임별 유사도 계산
        min_frames = min(len(keypoints1), len(keypoints2))
        similarities = []
        for i in range(min_frames):
            similarity = np.mean(np.abs(keypoints1[i] - keypoints2[i]))
            similarities.append(similarity)
        
        print("\n=== 유사도 분석 ===")
        print(f"평균 차이: {np.mean(similarities):.6f}")
        print(f"최대 차이: {np.max(similarities):.6f}")
        print(f"최소 차이: {np.min(similarities):.6f}")
        
        # 차이가 가장 큰 프레임 찾기
        max_diff_frame = np.argmax(similarities)
        print(f"\n가장 큰 차이가 나는 프레임: {max_diff_frame}")
        print("\n해당 프레임의 키포인트 비교 (처음 10개 값):")
        print(f"파일1: {keypoints1[max_diff_frame][:10]}")
        print(f"파일2: {keypoints2[max_diff_frame][:10]}")
        
        # 유사도 그래프 그리기
        plt.figure(figsize=(12, 6))
        plt.plot(similarities, label='프레임별 차이')
        plt.title('프레임별 키포인트 차이')
        plt.xlabel('프레임')
        plt.ylabel('평균 절대 차이')
        plt.grid(True)
        plt.legend()
        
        # 그래프 저장
        output_path = 'comparison_result.png'
        plt.savefig(output_path)
        print(f"\n그래프가 저장되었습니다: {output_path}")
        
    except Exception as e:
        print(f"에러 발생: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("사용법: python compare_npz.py <npz_파일1> <npz_파일2>")
        sys.exit(1)
    
    file1, file2 = sys.argv[1:3]
    if not Path(file1).exists() or not Path(file2).exists():
        print("파일을 찾을 수 없습니다.")
        sys.exit(1)
    
    compare_npz_files(file1, file2) 