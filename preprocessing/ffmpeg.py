import subprocess
import sys
from pathlib import Path


def convert_avi_to_wav(input_file, output_file=None):
    if output_file is None:
        output_file = Path(input_file).with_suffix('.wav')

    cmd = ['ffmpeg', '-y', '-i', input_file,
           '-acodec', 'pcm_s16le', str(output_file)]
    subprocess.run(cmd, check=True)
    print(f"변환 완료: {output_file}")


if __name__ == "__main__":
    convert_avi_to_wav(
        'C:/Users/USER/Documents/코골이/easy_sleep/PSG_sample.avi')
