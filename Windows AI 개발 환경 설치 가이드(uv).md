> 목적: Windows 환경에서 **Python 3.11 + uv + Cursor AI + LLM API** 기반 실습 환경을 안정적으로 구성하는 것을 목표로 함.

---

## 1. 기존 환경 정리 (기존 Anaconda, Python 사용자만)

> 다른 버전의 파이썬과 경로가 꼬여 발생하는 '실행 오류' 차단

### 1-1. Anaconda 제거

- Windows 아이콘 → **설정** → **앱** → **설치된 앱**
- Anaconda 옆 점 3개 → **제거**

### 1-2. 기존 Python 제거

- **Windows PowerShell** → **관리자 권한 실행**

### Python 설치 여부 체크

```powershell
where.exe python
```

- 결과에 Python 경로가 나오면 해당 파일 삭제
  - 만일 이렇게 나왔다면 … 
  - C:\Users\sangh\AppData\Local\Microsoft\WindowsApps\python.exe
  - 해당 경로 복사 후 (rm 경로 붙여넣기) 후 enter

```powershell
rm C:\Users\sangh\AppData\Local\Microsoft\WindowsApps\python.exe
```

---

## 2. Git 설치

> 오픈소스 코드를 내 컴퓨터로 내려받고 코드의 버전을 관리하기 위한 필수 도구

- 다운로드: [https://git-scm.com/download/win](https://git-scm.com/download/win)
- **64-bit Git for Windows Setup** 선택
- **설치 시 주의:** `Add a Git Bash Profile to Windows Terminal` 체크 확인.
- Next로 진행

설치 확인:

- Window 키 - PowerShell 을 반드시 **관리자 권한으로 실행**
- 아래의 명령어 "git" 을 입력하여 빨간 오류가 아닌 하얀 글씨가 잘 뜨는지 확인

```powershell
git
```

---

## 3. uv 설치

> 파이썬 설치, 라이브러리 관리, 가상환경 생성을 하나로 합친 올인원 도구

### 3-1. Home 디렉토리 이동

```powershell
cd ~
```

C:/Users/사용자ID 로 이동

### 3-2. Windows 실행 정책 변경

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser -Force
```

### 3-3. uv 설치

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3-4. **설치 확인**

- PowerShell 을 껐다 재실행 후 아래 명령어 입력

```PowerShell
uv --version
```

- uv 0.11.0 (1f31f0e9f 2026-03-23 x86_64-pc-windows-msvc) 등 결과 나오면 성공

## 4. Python 설치

```powershell
uv python install 3.11.9 //cd ~ 실행 후 홈 디렉토리에서 설치할 것.
```

- Python 설치 확인

```PowerShell
python --version

//혹시 안될 시 uv run python --version 으로 진행
```

---

## 5. 실습 코드 다운로드 및 환경 구성

### 5-1. 홈 디렉토리로 이동 및 프로젝트 다운로드

```powershell
cd ~
```

```PowerShell
git clone https://github.com/innocurveai/fintech.git
```

- 폴더로 이동

```powershell
cd fintech
```

### 5-2. 프로젝트 내 Python 버전 고정

```PowerShell
uv python pin 3.11.9
```

### 5-3. 가상환경 구축 및 패키지 설치

```PowerShell
uv sync
```

## 6. Cursor AI 설치

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

---

## 7. API Key 발급 및 설정

`.env-sample` → `.env` 로 파일명 변경

### 발급 대상

- OpenAI: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)

  - Copy하고 비밀 장소에 보관 (done 누르면 다시는 copy 할 수 없음)

> 모든 키는 `.env-sample` 파일에 입력하며,
> 입력 후 파일 이름을 `.env-sample` 에서 `.env` 로 변경 필수
> `fintech/` 에 위치해야 함
