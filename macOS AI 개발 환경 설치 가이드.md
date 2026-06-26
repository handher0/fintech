# macOS AI 개발 환경 설치 가이드

> 목적: macOS 환경에서 **Python 3.11 + uv + Cursor AI + LLM API** 기반 실습 환경을 안정적으로 구성하는 것을 목표로 함.

---

모든 명령어는 `터미널.app`에서 실행하시면 됩니다.
각 설치 과정에서 계속 진행할지를 묻는 단계가 나올 수 있습니다.
마지막 줄을 읽어 보시고, 상황에 맞게 `y`나 `yes`를 입력하신 후 엔터를 치시면 됩니다.

## 1. Homebrew 설치

Homebrew는 macOS 운영체제에서 다양한 라이브러리와 프로그램, 앱을 간편하게 설치하고 삭제할 수 있도록 도와주는 프로그램입니다.

```sh
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

## 2. uv 설치

uv는 Python과 pip 라이브러리를 버전에 맞게 프로젝트별로 설치하고 관리해 주는 프로그램입니다.

```sh
brew install uv
```

## 3. 실습 코드 다운로드 및 환경 구성

> 파이썬 설치, 라이브러리 관리, 가상환경 생성을 하나로 합친 올인원 도구

### 3-1. Home 디렉토리 이동

```powershell
cd ~
```

C:/Users/사용자ID 로 이동

### 3-2. 프로젝트 다운로드

```powershell
git clone https://github.com/innocurveai/fintech.git
```

- 폴더로 이동

```sh
cd fintech
```

### 3-3. 가상환경 및 패키지 설치

다음 명령으로 Python과 Python packages를 설치합니다.

```sh
uv sync
```

## 4. Cursor AI 설치

- 다운로드: [https://www.cursor.com/](https://www.cursor.com/)
- 설치 후 로그인 및 Extensions에서 아래 항목 설치

  - Extensions = 왼쪽 메뉴에서 테트리스 블록 모양 or ctrl+shift+X 로 실행
  - Python
    - ms-python 제공
  - Jupyter
    - ms-toolsai 제공
  - Office Viewer
    - cweijan 제공
  - vscode-pdf
    - tomoki1207
  - Korean Language Pack for Visual Studio Code
    - MS-CEINTL 제공, 지구본 아이콘

Cursor AI 껐다가 재실행



## 5. API Key 발급 및 설정

`.env-sample` → `.env` 로 파일명 변경

### 발급 대상

- OpenAI: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

  - Copy하고 비밀 장소에 보관 (done 누르면 다시는 copy 할 수 없음)

> 모든 키는 `.env-sample` 파일에 입력하며,
> 입력 후 파일 이름을 `.env-sample` 에서 `.env` 로 변경 필수
> `fintech/` 에 위치해야 함
