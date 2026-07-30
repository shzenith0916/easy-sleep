"""PSG 영상(.avi) → wav 추출.

입력 경로는 인자로 받는다. 환자 식별정보가 파일명에 들어갈 수 있으므로
경로를 소스에 하드코딩하지 말 것.
"""
import argparse
from pathlib import Path
import subprocess


def convert_avi_to_wav(input_file, output_file=None):
    if output_file is None:
        output_file = Path(input_file).with_suffix('.wav')

    cmd = ['ffmpeg', '-y', '-i', input_file,
           '-acodec', 'pcm_s16le', str(output_file)]
    subprocess.run(cmd, check=True)
    print(f"변환 완료: {output_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="PSG 영상(.avi) → wav 추출")
    ap.add_argument("input", help="입력 영상 경로")
    ap.add_argument("-o", "--output", default=None,
                    help="출력 wav 경로 (기본: 입력과 같은 이름의 .wav)")
    args = ap.parse_args()
    convert_avi_to_wav(args.input, args.output)
