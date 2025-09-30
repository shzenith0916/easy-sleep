import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter

# 예시: 1D 신호에서 median filter 적용


original = [1, 2, 100, 3, 4,]  # 이때 100은 스파이크 노이즈
# 3x3 윈도우에서 median 값으로 대체
filtered = [1, 2, 3, 3, 4]


def demonstrate_median_filter_effect(mel_spec_db):
    """미디안 필터 효과 확인"""

    # 다양한 필터 크기 테스트
    filter_sizes = [(1, 1),
                    (1, 3),
                    (3, 1),
                    (3, 3),
                    (5, 3),
                    (5, 5)]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    # 원본
    img0 = axes[0].imshow(mel_spec_db, aspect='auto', origin='lower')
    axes[0].set_title('Original Mel-spectrogram')
    plt.colorbar(img0, ax=axes[0])

    # 각 필터 크기별 효과
    for i, size in enumerate(filter_sizes, 1):  # 1부터 시작, 0은 original
        filtered = median_filter(mel_spec_db, size=size)
        img = axes[i].imshow(filtered, aspect='auto', origin='lower')
        axes[i].set_title(f"Median Filter {size}")
        plt.colorbar(img, ax=axes[i])

    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#   demonstrate_median_filter_effect(mel_spec_db)
