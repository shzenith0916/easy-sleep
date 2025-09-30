import os
import shutil
import re


def create_anonymized_copies(base_directory, anonymize_mode="mask", output_base_directory=None):
    """
    서브폴더에 있는 Video_Clip0001.avi 파일들을 익명화하여 새로운 폴더에 복사

    Args:
        base_directory (str): 기본 디렉토리 경로 (예: "4)severe")
        anonymize_mode (str): 익명화 모드 ("mask", "remove")
        output_base_directory (str): 출력 베이스 디렉토리 경로 (예: "C:\Users\USER\Desktop")
    """
    if not os.path.exists(base_directory):
        print(f"디렉토리 '{base_directory}'가 존재하지 않습니다.")
        return

    # 원본 폴더명 추출
    original_folder_name = os.path.basename(base_directory)

    # 출력 디렉토리 설정
    if output_base_directory is None:
        output_directory = f"{base_directory}_anonymized_{anonymize_mode}"
    else:
        output_directory = os.path.join(
            output_base_directory, f"{original_folder_name}_anonymized_{anonymize_mode}")

    # 새로운 익명화 폴더 생성
    os.makedirs(output_directory, exist_ok=True)

    print(f"'{base_directory}' 디렉토리를 스캔합니다...")
    print(f"익명화된 파일들이 '{output_directory}' 폴더에 저장됩니다.")

    # 복사된 파일 목록 저장
    copied_files = []

    # 서브폴더 탐색
    for item in os.listdir(base_directory):
        subfolder_path = os.path.join(base_directory, item)

        # 서브폴더인지 확인
        if os.path.isdir(subfolder_path):
            print(f"\n서브폴더 '{item}' 처리 중...")

            # 새로운 서브폴더 생성
            anonymized_subfolder = os.path.join(output_directory, item)
            os.makedirs(anonymized_subfolder, exist_ok=True)

            # 서브폴더 내의 .avi 파일 찾기
            for file in os.listdir(subfolder_path):
                if file.endswith('.avi'):
                    old_file_path = os.path.join(subfolder_path, file)

                    # 사람 이름 추출 (폴더명에서 특수문자 제거하고 정리)
                    person_name = extract_person_name(item, anonymize_mode)

                    # 새로운 파일명 생성
                    new_filename = f"{person_name}_{file}"
                    new_file_path = os.path.join(
                        anonymized_subfolder, new_filename)

                    # 파일 복사
                    try:
                        shutil.copy2(old_file_path, new_file_path)
                        copied_files.append((old_file_path, new_file_path))
                        print(f"  '{file}' -> '{new_filename}' (복사 완료)")
                    except Exception as e:
                        print(f"  오류: {file} 복사 실패 - {e}")

    # 결과 요약
    print(f"\n=== 작업 완료 ===")
    print(f"총 {len(copied_files)}개 파일이 익명화되어 복사되었습니다.")
    print(f"원본 폴더: {base_directory}")
    print(f"익명화 폴더: {output_directory}")

    if copied_files:
        print("\n복사된 파일 목록:")
        for old_path, new_path in copied_files:
            print(f"  {os.path.basename(old_path)} -> {os.path.basename(new_path)}")


def extract_person_name(folder_name, anonymize_mode="mask"):
    """
    폴더명에서 사람 이름을 추출하고 익명화하는 함수

    Args:
        folder_name (str): 폴더 이름
        anonymize_mode (str): 익명화 모드
            - "mask": 김철수 -> 김*수 (중간 글자 마스킹)
            - "remove": 이름 완전 제거

    Returns:
        str: 추출된 사람 이름 (익명화된 형태)
    """
    ## 특수문자 제거 및 정리 ##
    # 괄호, 숫자, 특수문자 등을 제거하여 사람 이름만 추출
    name = re.sub(r'[^\w\s가-힣]', '', folder_name)  # 한글, 영문, 숫자, 공백만 유지

    # 숫자 제거 - 환자코드가 파일명에 포함될 때
    # name = re.sub(r'\d+', '', name)

    # 앞뒤 공백 제거
    name = name.strip()

    # 파일명 사이 공백을 언더스코어로 변경
    name = name.replace(' ', '_')

    if not name:
        return folder_name  # 이름이 없으면 원본 폴더명 사용

    # 익명화 처리
    if anonymize_mode == "mask":
        # 한글 이름 마스킹 (김철수 -> 김*수)
        if len(name) >= 2:
            if len(name) == 2:
                # 2글자 이름: 김수 -> 김*
                masked_name = name[0] + "*"
            else:
                # 3글자 이상: 김철수 -> 김*수
                masked_name = name[0] + "*" + name[-1]
        else:
            masked_name = name
        return masked_name

    elif anonymize_mode == "remove":
        # 이름 완전 제거, 폴더 번호나 순서로 대체
        return "Person"


def preview_changes(base_directory, anonymize_mode="mask", output_base_directory=None):
    """
    실제 복사 전에 어떤 파일들이 어떻게 복사될지 미리 보여주는 함수

    Args:
        base_directory (str): 원본 디렉토리 경로
        anonymize_mode (str): 익명화 모드 ("mask", "remove")
        output_base_directory (str): 출력 베이스 디렉토리 경로 (선택사항)
    """
    if not os.path.exists(base_directory):
        print(f"디렉토리 '{base_directory}'가 존재하지 않습니다.")
        return

    # 원본 폴더명 추출
    original_folder_name = os.path.basename(base_directory)

    if output_base_directory is None:
        output_directory = f"{base_directory}_anonymized_{anonymize_mode}"
    else:
        output_directory = os.path.join(
            output_base_directory, f"{original_folder_name}_anonymized_{anonymize_mode}")

    print(f"'{base_directory}' 속 파일 복사 미리보기...")
    print(f"새로운 폴더: '{output_directory}'")

    changes = []

    for item in os.listdir(base_directory):
        subfolder_path = os.path.join(base_directory, item)

        if os.path.isdir(subfolder_path):
            print(f"\n서브폴더: {item}")

            for file in os.listdir(subfolder_path):
                if file.endswith('.avi'):
                    person_name = extract_person_name(item, anonymize_mode)
                    new_filename = f"{person_name}_{file}"

                    print(f"파일 복사: {file} -> {new_filename}")
                    changes.append((item, file, new_filename))

    print(f"\n총 {len(changes)}개 파일이 복사될 예정입니다.")
    print(f"원본 폴더는 그대로 유지되고, '{output_directory}' 폴더에 익명화된 복사본이 생성됩니다.")
    return changes


if __name__ == "__main__":

    # 사용자에게 원본 폴더 경로 입력 요청
    print("익명화할 원본 폴더의 경로를 입력해주세요:")
    base_dir = input("폴더 경로: ").strip()

    # 빈 입력 처리
    if not base_dir:
        print("폴더 경로가 입력되지 않았습니다. 프로그램을 종료합니다.")
        exit()

    # 폴더 존재 여부 확인
    if not os.path.exists(base_dir):
        print(f"입력하신 폴더 '{base_dir}'가 존재하지 않습니다. 프로그램을 종료합니다.")
        exit()

    # 출력 베이스 디렉토리 입력
    print(f"\n익명화된 폴더를 저장할 위치를 입력해주세요:")
    print("저장할 베이스 디엑토리 경로 입력 예시: C:\\Users\\USER\\Desktop 또는 빈칸으로 두면 원본 폴더와 같은 위치에 저장)")
    output_base_dir = input("저장 위치: ").strip()

    # 빈 입력이면 None으로 설정 (원본 폴더와 같은 위치)
    if not output_base_dir:
        output_base_dir = None
    else:
        # 출력 디렉토리 존재 여부 확인
        if not os.path.exists(output_base_dir):
            print(f"입력하신 저장 위치 '{output_base_dir}'가 존재하지 않습니다. 프로그램을 종료합니다.")
            exit()

    # 익명화 모드 선택
    print("\n개인정보 보호를 위한 익명화 모드를 선택하세요:")
    print("1. 마스킹 (김철수 -> 김*수)")
    print("2. 완전 제거 (김철수 -> Person)")

    while True:
        choice = input("선택 (1 or 2): ").strip()
        if choice == "1":
            anonymize_mode = "mask"
            break
        elif choice == "2":
            anonymize_mode = "remove"
            break
        else:
            print("1 또는 2를 입력해주세요. 혹은 Ctrl+C를 눌러 중단합니다.")

    # 미리보기 실행
    print(f"\n복사 사항을 미리 확인합니다. 익명화 모드: {anonymize_mode}\n")
    preview_changes(base_dir, anonymize_mode, output_base_dir)

    # 사용자 확인
    print("\n위 내용으로 익명화된 복사본을 생성하시겠습니까? (y/n): ", end="")
    confirm = input().strip().lower()

    if confirm in ['y', 'yes']:
        # 실제 파일 복사 실행
        create_anonymized_copies(base_dir, anonymize_mode, output_base_dir)
    else:
        print("작업이 취소되었습니다.")
