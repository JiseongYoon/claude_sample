# Local LLM AI Agent

로컬 GPU 위에서 구동되는 헤드리스 API-퍼스트 AI 에이전트 플랫폼입니다.
Gemma-4 31B 모델을 llama.cpp로 서빙하고, 모듈형 기능(문서 QA, 원격 스토리지,
웹 브라우저, 샌드박스 실행, MCP 도구, 비전/멀티모달)을 안전 게이트 아래에서
에이전트 루프로 조합합니다. React Web UI가 별도 포함됩니다.

---

## 목차

1. [하드웨어 요구사항](#1-하드웨어-요구사항)
2. [소프트웨어 사전 준비](#2-소프트웨어-사전-준비)
3. [Conda 환경 생성](#3-conda-환경-생성)
4. [모델 파일 준비](#4-모델-파일-준비)
5. [프로젝트 설치](#5-프로젝트-설치)
6. [환경 변수 (.env) 설정](#6-환경-변수-env-설정)
7. [llama-server (모델 서버) 실행](#7-llama-server-모델-서버-실행)
8. [API 게이트웨이 서버 실행](#8-api-게이트웨이-서버-실행)
9. [Web UI 빌드 및 실행](#9-web-ui-빌드-및-실행)
10. [인증 및 토큰 발급](#10-인증-및-토큰-발급)
11. [기능별 활성화 가이드](#11-기능별-활성화-가이드)
12. [Docker 샌드박스 (exec) 설정](#12-docker-샌드박스-exec-설정)
13. [비전/멀티모달 설정](#13-비전멀티모달-설정)
14. [API 엔드포인트 요약](#14-api-엔드포인트-요약)
15. [테스트 실행](#15-테스트-실행)
16. [스모크 테스트 (실제 모델)](#16-스모크-테스트-실제-모델)
17. [문제 해결](#17-문제-해결)
18. [아키텍처 개요](#18-아키텍처-개요)

---

## 1. 하드웨어 요구사항

| 항목 | 최소 사양 |
|------|----------|
| OS | WSL2 Linux (Windows 11 권장) |
| GPU | NVIDIA GPU 1장 이상, 합산 VRAM ≥ 48GB (Q6 양자화 기준 ~36GB 사용) |
| RAM | 32GB 이상 권장 |
| 디스크 | 모델 파일 포함 ~100GB 여유 공간 |
| CUDA | CUDA 13.0+ (conda-forge llama.cpp가 자체 CUDA 런타임 포함) |
| Docker | `enable_exec` 사용 시 Docker Desktop / Docker Engine 필수 |

> **GPU 참고**: `CUDA_VISIBLE_DEVICES`로 사용할 GPU 인덱스를 지정하세요.
> 예: GPU 2장 → `CUDA_VISIBLE_DEVICES=0,1`, 1장 → `CUDA_VISIBLE_DEVICES=0`.

---

## 2. 소프트웨어 사전 준비

다음이 시스템에 설치되어 있어야 합니다:

```bash
# 1. Anaconda 또는 Miniconda
#    https://www.anaconda.com/download 에서 설치
#    설치 후 쉘에서 conda 명령이 작동하는지 확인:
conda --version

# 2. NVIDIA 드라이버 확인 (WSL2에서)
nvidia-smi
#    GPU 2장이 표시되는지, 드라이버 버전이 535+ 이상인지 확인

# 3. Git
git --version

# 4. Docker (enable_exec 기능 사용 시에만 필요)
docker --version
```

---

## 3. Conda 환경 생성

전용 conda 환경을 만들고 llama.cpp CUDA 빌드를 설치합니다.
**모든 작업은 이 환경 안에서 수행해야 합니다** — 절대 base나 글로벌 환경을 사용하지 마세요.

```bash
# 3-1. 환경 생성 (Python 3.11 + Node.js 22 LTS)
conda create -y -n local-ai-agent-env-1 python=3.11 "nodejs=22" -c conda-forge

# 3-2. 환경 활성화
source ~/anaconda3/etc/profile.d/conda.sh
conda activate local-ai-agent-env-1

# 3-3. 활성화 확인
python --version    # → Python 3.11.x
node --version      # → v22.x.x
npm --version       # → 10.x.x

# 3-4. llama.cpp 설치 (CUDA 빌드 — GPU 서빙에 필수)
conda install -y -c conda-forge "llama.cpp=*=cuda130*"

# 3-5. llama-server 설치 확인
llama-server --version
```

> **환경 위치**: `/home/<사용자>/anaconda3/envs/local-ai-agent-env-1`
>
> **중요**: 이후 모든 명령에서 이 환경이 활성화된 상태여야 합니다.
> 새 터미널을 열 때마다 아래를 실행하세요:
> ```bash
> source ~/anaconda3/etc/profile.d/conda.sh && conda activate local-ai-agent-env-1
> ```

---

## 4. 모델 파일 준비

이 프로젝트는 **Gemma-4 31B** 모델의 GGUF 양자화 파일을 사용합니다.
**모델 파일은 배포 패키지에 포함되어 있지 않습니다.** 별도로 다운로드하여 원하는 위치에 배치한 뒤,
`.env`의 경로 변수를 해당 위치로 지정해야 합니다.

### 4-1. 디렉터리 준비

모델을 저장할 디렉터리를 만듭니다. 위치는 자유이며, `.env`에서 경로를 지정합니다.

```bash
# 예시: 프로젝트 내부에 만들 경우
mkdir -p models/gemma-4-gguf
mkdir -p models/gemma-4-safetensors

# 예시: 외부 대용량 디스크를 사용할 경우
# mkdir -p /data/models/gemma-4-gguf
# mkdir -p /data/models/gemma-4-safetensors
```

최종 구조 (경로는 예시):
```
<MODEL_GGUF_DIR>/                          # .env의 MODEL_GGUF_DIR이 가리키는 경로
├── gemma-4-31B-it-UD-Q6_K_XL.gguf        # 기본 서빙 파일 (GGUF_FILE)
├── gemma-4-31B-it-UD-Q4_K_XL.gguf        # (선택) 가벼운 대안
└── gemma-4-31B-it-UD-Q8_K_XL.gguf        # (선택) 무거운 대안

<MODEL_SAFETENSORS_DIR>/                   # .env의 MODEL_SAFETENSORS_DIR이 가리키는 경로
└── (Hugging Face 원본 safetensors)        # 비전/멀티모달 프로젝터 생성 시에만 필요
```

### 4-2. 모델 다운로드

모델 파일은 직접 다운로드해야 합니다:
- **GGUF 파일** (필수): Hugging Face에서 Gemma-4 31B GGUF 양자화 파일을 다운로드하여 `MODEL_GGUF_DIR`에 배치
- **Safetensors 파일** (선택 — 비전 프로젝터 생성 시에만 필요): Hugging Face에서 원본 safetensors를 다운로드하여 `MODEL_SAFETENSORS_DIR`에 배치

### 4-3. 권장 양자화 선택

| 양자화 | VRAM 사용량 | 생성 속도 | 용도 |
|--------|-----------|----------|------|
| Q4_K_XL | ~24GB | ~30.7 tok/s | 빠른 추론 우선 |
| **Q6_K_XL** | **~36GB** | **~22 tok/s** | **기본값 (품질/속도 균형)** |
| Q8_K_XL | ~46GB | ~18.4 tok/s | 최고 품질 |

---

## 5. 프로젝트 설치

```bash
# 5-1. 프로젝트 디렉터리로 이동
cd /path/to/local_ai_agent

# 5-2. conda 환경 활성화 (아직 안 했다면)
source ~/anaconda3/etc/profile.d/conda.sh && conda activate local-ai-agent-env-1

# 5-3. 코어 패키지 설치 (editable mode + 개발 의존성)
pip install -e ".[dev]"

# 5-4. 설치 확인 — console script가 등록되었는지 확인
local-ai-agent --help 2>&1 || echo "설치 확인 완료 (help 미지원 시 정상)"

# 5-5. (선택) 사용할 기능에 따라 추가 의존성 설치
#      활성화하지 않을 기능은 설치할 필요 없습니다.
pip install -e ".[docqa]"        # 문서 QA (enable_docqa)
pip install -e ".[storage]"      # 원격 스토리지 SSH/SFTP (enable_storage)
pip install -e ".[browser]"      # 웹 브라우저/검색 (enable_browser)
pip install -e ".[mcp]"          # MCP 외부 도구 (enable_mcp)
pip install -e ".[multimodal]"   # 비전/멀티모달 (비전 프로젝터 필요)

# 5-6. Web UI 의존성 설치
npm --prefix web install
```

---

## 6. 환경 변수 (.env) 설정

프로젝트 루트에 `.env` 파일을 생성합니다. **이 파일은 절대 Git에 커밋하지 마세요.**

```bash
# .env 파일 생성 (프로젝트 루트에서)
cat > .env << 'EOF'
# ============================================================
# Local AI Agent — 환경 변수 설정
# ============================================================

# --- 모델 경로 (필수 — 기본값 없음, 실제 모델 위치로 수정하세요) ---
MODEL_SAFETENSORS_DIR=/path/to/your/gemma-4-safetensors
MODEL_GGUF_DIR=/path/to/your/gemma-4-gguf
GGUF_FILE=gemma-4-31B-it-UD-Q6_K_XL.gguf

# --- 서빙 엔진 ---
SERVE_ENGINE=llamacpp
SERVE_HOST=127.0.0.1
SERVE_PORT=8000
SERVED_MODEL_NAME=gemma-4-31b-it

# --- GPU 설정 ---
CUDA_VISIBLE_DEVICES=0,1
N_GPU_LAYERS=999
SPLIT_MODE=layer
TENSOR_SPLIT=1,1
CTX_SIZE=16384
USE_JINJA=true

# --- API 게이트웨이 ---
API_PORT=8080

# --- 보안 (필수 — 직접 생성한 강력한 값을 넣으세요) ---
AUTH_ENABLED=true
API_KEY=여기에-강력한-랜덤-키를-넣으세요
JWT_SECRET=여기에-다른-강력한-랜덤-값을-넣으세요

# --- CORS / 호스트 ---
# Web UI를 사용할 경우 UI 오리진을 반드시 추가
CORS_ALLOW_ORIGINS=["http://127.0.0.1:5173","http://localhost:5173"]
TRUSTED_HOSTS=["127.0.0.1","localhost","testserver"]

# --- 프로덕션 보안 ---
EXPOSE_HEALTH_DETAIL=false

# --- 기능 활성화 (필요한 것만 true로) ---
ENABLE_MODEL_STACK=true
ENABLE_AGENT=false
ENABLE_DOCQA=false
ENABLE_STORAGE=false
ENABLE_BROWSER=false
ENABLE_EXEC=false
ENABLE_MCP=false
ENABLE_PRESENTATION_DEMO=false
ENABLE_DEMO_MODULES=false
EOF
```

### 6-1. API_KEY와 JWT_SECRET 생성

**반드시 강력한 랜덤 값을 사용하세요:**

```bash
# API_KEY 생성
python -c "import secrets; print(secrets.token_urlsafe(48))"
# 출력 예시: aB3xF7...kL9mN2 (64자)

# JWT_SECRET 생성 (API_KEY와 다른 값)
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

생성된 값을 `.env` 파일의 `API_KEY`와 `JWT_SECRET`에 각각 붙여넣으세요.

### 6-2. 포트 설정 주의사항

| 포트 | 용도 | 설정 변수 | 기본값 |
|------|------|----------|--------|
| 8000 | 모델 서버 (llama-server) | `SERVE_PORT` | 8000 |
| 8080 | API 게이트웨이 (uvicorn) | `API_PORT` | 8080 |
| 5173 | Web UI (Vite dev server) | — | 5173 |

> **핵심**: `SERVE_PORT`와 `API_PORT`는 **반드시 다른 값**이어야 합니다.
> 같은 값을 넣으면 시작 시 검증 에러가 발생합니다.

---

## 7. llama-server (모델 서버) 실행

모델 서버를 먼저 시작합니다. `ENABLE_MODEL_STACK=true`일 때 API 게이트웨이의
모델 매니저가 자동으로 관리하지만, 수동으로 직접 시작할 수도 있습니다.

### 7-1. 수동 실행 (직접 테스트용)

```bash
# conda 환경 활성화 확인
source ~/anaconda3/etc/profile.d/conda.sh && conda activate local-ai-agent-env-1

# llama-server 실행 (-m 경로를 실제 GGUF 파일 위치로 수정하세요)
CUDA_VISIBLE_DEVICES=0,1 llama-server \
  -m /path/to/your/gemma-4-gguf/gemma-4-31B-it-UD-Q6_K_XL.gguf \
  -ngl 999 \
  --split-mode layer \
  -ts 1,1 \
  -c 16384 \
  --jinja \
  --alias gemma-4-31b-it \
  --port 8000
```

### 7-2. 실행 옵션 설명

| 옵션 | 설명 |
|------|------|
| `-m` | GGUF 모델 파일 경로 |
| `-ngl 999` | 모든 레이어를 GPU에 오프로드 |
| `--split-mode layer` | 레이어 단위로 GPU 분할 |
| `-ts 1,1` | 두 GPU에 균등 분배 |
| `-c 16384` | 컨텍스트 윈도우 크기 |
| `--jinja` | **필수** — Gemma-4 채팅 템플릿 올바르게 처리 |
| `--alias` | OpenAI 호환 API에서 사용할 모델 이름 |
| `--port` | 바인딩 포트 (기본 8000) |

### 7-3. 실행 확인

```bash
# 모델 서버가 준비되었는지 확인
curl http://127.0.0.1:8000/health
# → {"status":"ok"} 이면 정상

# OpenAI 호환 API 테스트
curl http://127.0.0.1:8000/v1/models
# → 모델 목록이 출력되면 정상
```

> **참고**: `ENABLE_MODEL_STACK=true`로 API 게이트웨이를 실행하면 모델 매니저가
> llama-server를 자동으로 시작/중지/전환합니다. 이 경우 수동 실행은 불필요합니다.

---

## 8. API 게이트웨이 서버 실행

```bash
# conda 환경 활성화 확인
source ~/anaconda3/etc/profile.d/conda.sh && conda activate local-ai-agent-env-1

# 프로젝트 루트로 이동
cd /path/to/local_ai_agent

# 서버 시작 (console script 사용)
local-ai-agent
```

정상 시작 시 로그:
```
YYYY-MM-DD HH:MM:SS INFO uvicorn: Uvicorn running on http://127.0.0.1:8080
```

### 8-1. 시작 확인

```bash
# 헬스 체크 (인증 불필요 — 공개 엔드포인트)
curl http://127.0.0.1:8080/health
# → {"status":"healthy", ...}

# 기능 목록 확인 (인증 필요)
curl -H "X-API-Key: <YOUR_API_KEY>" http://127.0.0.1:8080/capabilities
```

### 8-2. 프로세스 관리

```bash
# 포그라운드 실행 (Ctrl+C로 종료)
local-ai-agent

# 백그라운드 실행
nohup local-ai-agent > /tmp/local-ai-agent.log 2>&1 &

# 로그 확인
tail -f /tmp/local-ai-agent.log
```

---

## 9. Web UI 빌드 및 실행

Web UI는 별도의 React 앱(`web/` 디렉터리)이며, 코어 API와 독립적으로 동작합니다.
**Web UI는 zero-authority 클라이언트**입니다 — 모든 권한 판단은 서버 게이트에서 수행됩니다.

### 9-1. 개발 서버 (개발/테스트용)

```bash
# 의존성 설치 (최초 1회)
npm --prefix web install

# 개발 서버 시작 (핫 리로드)
npm --prefix web run dev
# → http://localhost:5173 에서 접속
```

### 9-2. 프로덕션 빌드

```bash
# 타입 체크 + 빌드
npm --prefix web run build

# 빌드 결과 확인
ls web/dist/
# → index.html, assets/ 등

# 빌드 결과 미리보기
npm --prefix web run preview
# → http://localhost:4173 에서 접속
```

### 9-3. CORS 설정 (필수)

Web UI가 API 게이트웨이에 접근하려면 `.env`의 `CORS_ALLOW_ORIGINS`에
UI의 오리진을 반드시 추가해야 합니다:

```bash
# 개발 서버 사용 시
CORS_ALLOW_ORIGINS=["http://127.0.0.1:5173","http://localhost:5173"]

# 프로덕션 배포 시 (실제 도메인으로 교체)
CORS_ALLOW_ORIGINS=["https://your-domain.com"]
```

> **경고**: 와일드카드(`*`)는 절대 사용하지 마세요. 명시적 오리진 목록만 허용됩니다.

### 9-4. Web UI 사용법

1. 브라우저에서 `http://localhost:5173` 접속
2. **Settings** 영역에서:
   - **API URL**: `http://127.0.0.1:8080` (API 게이트웨이 주소)
   - **API Key**: `.env`에 설정한 `API_KEY` 값 입력
   - **Auth Scheme**: `api-key` 선택
3. 연결되면 모델 로드/채팅/에이전트 기능 사용 가능

---

## 10. 인증 및 토큰 발급

### 10-1. 인증 방식

| 방식 | 용도 | 헤더/파라미터 |
|------|------|-------------|
| API Key | REST API 직접 호출 (cURL, 스크립트) | `X-API-Key: <key>` 헤더 |
| JWT | WebSocket 연결 (브라우저 Web UI) | `?token=<jwt>` 쿼리 파라미터 |

### 10-2. JWT 토큰 발급 (Mint Flow)

브라우저 WebSocket은 커스텀 헤더를 보낼 수 없으므로, 먼저 API Key로 JWT를 발급받습니다:

```bash
# JWT 발급 (Mint Flow)
curl -X POST http://127.0.0.1:8080/auth/token \
  -H "X-API-Key: <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"scopes": ["chat", "model", "ingest"]}'
# → {"token": "eyJhbGciOiJI..."}
```

사용 가능한 스코프: `chat`, `model`, `ingest`, `admin` 등 (기능에 따라 다름).

### 10-3. WebSocket 연결

```
ws://127.0.0.1:8080/ws?token=<JWT_TOKEN>
```

> **보안 주의**: URL에 API Key를 직접 넣지 마세요 (액세스 로그에 노출됨).
> WebSocket은 반드시 JWT만 사용합니다. API Key를 URL `?token=`에 넣으면
> 1008 코드로 연결이 거부됩니다.

---

## 11. 기능별 활성화 가이드

모든 기능은 **기본 OFF**입니다. `.env`에서 필요한 것만 `true`로 설정하세요.

### 11-1. 모델 스택 (`ENABLE_MODEL_STACK=true`)

모델 로드/언로드/전환 + 채팅/파라미터 튜닝 기능.

```bash
# .env에 추가 (이미 설정된 값 확인)
ENABLE_MODEL_STACK=true
MODEL_SAFETENSORS_DIR=/path/to/your/gemma-4-safetensors   # .env와 동일 경로
MODEL_GGUF_DIR=/path/to/your/gemma-4-gguf                 # .env와 동일 경로
GGUF_FILE=gemma-4-31B-it-UD-Q6_K_XL.gguf
```

추가 의존성: 없음 (llama-server는 conda로 이미 설치됨)

### 11-2. 에이전트 (`ENABLE_AGENT=true`)

도구 호출 기반 에이전트 루프 (HITL 승인 게이트 포함).

```bash
ENABLE_AGENT=true
ENABLE_MODEL_STACK=true    # 에이전트는 모델 스택 필요
```

추가 의존성: 없음

### 11-3. 문서 QA (`ENABLE_DOCQA=true`)

로컬 문서 질의응답/요약 + 파일 업로드(ingest).

```bash
ENABLE_DOCQA=true
ENABLE_MODEL_STACK=true    # 채팅 모델 필요
DOCS_ROOT=/path/to/documents   # 문서 QA가 읽을 루트 디렉터리
```

```bash
pip install -e ".[docqa]"
```

### 11-4. 원격 스토리지 (`ENABLE_STORAGE=true`)

SSH/SFTP를 통한 원격 파일 접근.

```bash
ENABLE_STORAGE=true
STORAGE_CONNECTORS_FILE=configs/storage_connectors.json
```

커넥터 설정 파일 형식:
```json
{
  "connectors": [
    {
      "name": "my-server",
      "kind": "ssh",
      "host": "192.168.1.100",
      "port": 22,
      "username": "user",
      "auth": {
        "key_path": "/home/user/.ssh/id_rsa"
      },
      "allowed_root": "/data",
      "read_only": true,
      "max_bytes": 10000000,
      "known_hosts_path": "/home/user/.ssh/known_hosts"
    }
  ]
}
```

> `auth`는 `key_path` 또는 `password_env` 중 **하나만** 사용합니다.
> `password_env`는 비밀번호가 담긴 **호스트 환경 변수의 이름**입니다 (값 자체가 아님).

```bash
pip install -e ".[storage]"
```

### 11-5. 웹 브라우저 (`ENABLE_BROWSER=true`)

웹 검색 + 페이지 내용 추출 (SearXNG 필요).

```bash
ENABLE_BROWSER=true
ENABLE_MODEL_STACK=true              # web_answer 기능에 필요
BROWSER_SEARXNG_URL=http://localhost:8888   # SearXNG 인스턴스 URL
```

```bash
pip install -e ".[browser]"
```

### 11-6. 샌드박스 실행 (`ENABLE_EXEC=true`)

Docker 컨테이너 안에서 격리된 명령 실행. [12장](#12-docker-샌드박스-exec-설정) 참조.

```bash
ENABLE_EXEC=true
```

추가 의존성: Docker (pip 패키지 불필요 — CLI subprocess 사용)

### 11-7. MCP 외부 도구 (`ENABLE_MCP=true`)

외부 MCP 서버의 도구를 에이전트 도구로 게이트 경유 사용.

```bash
ENABLE_MCP=true
MCP_SERVERS_FILE=configs/mcp_servers.json
```

서버 설정 파일 형식:
```json
{
  "servers": [
    {
      "name": "my-tool-server",
      "command": "npx",
      "args": ["-y", "@example/mcp-server"],
      "env": {
        "NON_SECRET_VAR": "value"
      },
      "secret_env": ["MY_API_TOKEN"]
    }
  ]
}
```

> `secret_env`는 비밀이 담긴 **호스트 환경 변수 이름 목록**입니다.
> `env`에 비밀처럼 보이는 키를 넣으면 거부됩니다 — `secret_env`를 사용하세요.

```bash
pip install -e ".[mcp]"
```

---

## 12. Docker 샌드박스 (exec) 설정

`enable_exec` 기능은 Docker 컨테이너 내에서 격리된 코드/명령 실행을 제공합니다.
**보안에 직접적인 영향을 미치는 설정입니다 — 주의해서 구성하세요.**

### 12-1. 하드닝된 샌드박스 이미지 빌드

```dockerfile
# Dockerfile.exec (예시 — 프로젝트 요구에 맞게 조정)
FROM ubuntu:22.04

# 비-root 사용자 생성 (UID:GID = 1000:1000)
RUN groupadd -g 1000 sandbox && useradd -u 1000 -g 1000 -m sandbox

# 워크스페이스 디렉터리 생성 (sandbox 사용자 소유)
RUN mkdir -p /workspace && chown 1000:1000 /workspace

# 필요한 런타임만 최소 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

USER sandbox
WORKDIR /workspace
```

```bash
# 이미지 빌드 (이름은 반드시 local_ai_agent_exec:base)
docker build -t local_ai_agent_exec:base -f Dockerfile.exec .
```

### 12-2. 보안 하드닝 (자동 적용)

서버가 컨테이너 실행 시 자동으로 적용하는 보안 조치:

| 보호 | 설정 |
|------|------|
| 네트워크 | `network=none` (외부 통신 완전 차단) |
| 루트 파일시스템 | 읽기 전용 (read-only) |
| 리눅스 캐퍼빌리티 | 모두 제거 (cap-drop ALL) |
| 권한 상승 | 차단 (no-new-privileges) |
| 메모리 | 2GB 제한 (스왑 탈출 불가) |
| PID | 256개 제한 (포크 폭탄 방지) |
| 사용자 | 비-root (1000:1000) |

---

## 13. 비전/멀티모달 설정

이미지/스캔 PDF를 이해하는 비전 기능은 **mmproj 비전 프로젝터 GGUF**가 있어야 활성화됩니다.
프로젝터는 리포지토리에 포함되어 있지 않으며, safetensors에서 직접 생성해야 합니다.

### 13-1. 비전 프로젝터 생성

```bash
# llama.cpp 소스에서 변환 스크립트 사용
python llama.cpp/convert_hf_to_gguf.py /path/to/your/gemma-4-safetensors --mmproj \
  --outfile /path/to/your/gemma-4-gguf/gemma-4-mmproj-f16.gguf
```

### 13-2. 환경 변수 설정

```bash
# .env에 추가
MODEL_MMPROJ_FILE=gemma-4-mmproj-f16.gguf   # MODEL_GGUF_DIR 안의 파일명
ENABLE_DOCQA=true                             # 비전은 DocQA 기반
```

### 13-3. 추가 의존성

```bash
pip install -e ".[multimodal]"   # pymupdf — 스캔 PDF 래스터라이제이션
pip install -e ".[docqa]"        # DocQA도 함께 필요
```

> 프로젝터가 설정되면 llama-server가 `--mmproj` 옵션으로 자동 시작되며,
> `multimodal` 기능이 활성화되고, ingest 엔드포인트가 이미지 타입(png/jpg/jpeg/webp)을
> 추가로 수락합니다. 프로젝터가 없으면 텍스트 전용으로 정상 동작합니다.

---

## 14. API 엔드포인트 요약

| 엔드포인트 | 메서드 | 인증 | 설명 |
|-----------|--------|------|------|
| `/health` | GET | 불필요 | 서버 상태 확인 (프로덕션에서는 내부 상세 숨김) |
| `/capabilities` | GET | 필요 | 활성화된 기능 목록 |
| `/auth/token` | POST | API Key | JWT 토큰 발급 (Mint Flow) |
| `/model/load` | POST | 필요 | 모델 로드 |
| `/model/unload` | POST | 필요 | 모델 언로드 |
| `/model/status` | GET | 필요 | 모델 상태 확인 |
| `/v1/chat/completions` | POST | 필요 | 채팅 (OpenAI 호환) |
| `/model/params` | GET/POST | 필요 | 파라미터 조회/수정 |
| `/ws` | WebSocket | JWT (`?token=`) | 실시간 양방향 통신 (에이전트, 채팅 스트리밍) |

---

## 15. 테스트 실행

### 15-1. Python 테스트 (1254개)

```bash
# conda 환경 활성화 확인
source ~/anaconda3/etc/profile.d/conda.sh && conda activate local-ai-agent-env-1

# 전체 테스트 실행
pytest

# 특정 모듈만 실행
pytest tests/test_config.py
pytest tests/test_gateway.py

# 상세 출력
pytest -v
```

### 15-2. Web UI 테스트 (157개 Vitest)

```bash
# 단일 실행
npm --prefix web test

# 감시 모드 (파일 변경 시 자동 재실행)
npm --prefix web run test:watch

# 타입 체크
npm --prefix web run lint
```

### 15-3. E2E 테스트 (16개)

실제 API를 거치는 통합 E2E 테스트:

```bash
# 프로젝트 루트에서 실행
bash web/e2e/run_e2e.sh
```

> E2E 테스트는 자체적으로 가짜 코어 서버를 시작하므로 실제 서버가 실행 중이지 않아도 됩니다.

---

## 16. 스모크 테스트 (실제 모델)

실제 모델과 GPU가 필요한 엔드-투-엔드 검증 스크립트입니다.
**서버가 실행 중인 상태에서 별도 터미널에서 실행하세요.**

```bash
# 에이전트 루프 스모크 테스트 (모델이 도구 호출 → 게이트 → 디스패치 → 스트리밍 응답)
python scripts/smoke_agent_loop.py

# 문서 QA 스모크 테스트
python scripts/smoke_docqa.py

# 브라우저 스모크 테스트
python scripts/smoke_browser.py

# 스토리지 스모크 테스트
python scripts/smoke_storage.py

# 실행(exec) 스모크 테스트
python scripts/smoke_exec.py

# MCP 스모크 테스트
python scripts/smoke_mcp.py
```

---

## 17. 문제 해결

### Q: `local-ai-agent` 명령이 안 됩니다
```bash
# conda 환경 활성화 확인
conda activate local-ai-agent-env-1
# editable 설치 확인
pip install -e ".[dev]"
```

### Q: 서버 시작 시 `api_port must differ from serve_port` 에러
`.env`에서 `API_PORT`와 `SERVE_PORT`가 서로 다른 값인지 확인하세요.
기본값: API_PORT=8080, SERVE_PORT=8000.

### Q: 모델 로드 후 채팅이 안 됩니다
```bash
# 모델 서버 직접 테스트
curl http://127.0.0.1:8000/v1/models
# llama-server에 --jinja 옵션이 있는지 확인 (Gemma-4 필수)
```

### Q: Web UI에서 API 연결이 안 됩니다
1. `.env`의 `CORS_ALLOW_ORIGINS`에 UI 오리진이 포함되어 있는지 확인
2. API Key가 올바른지 확인
3. 서버가 실행 중인지 확인: `curl http://127.0.0.1:8080/health`

### Q: WebSocket 연결 시 `auth_failed` (1008)
WebSocket은 JWT만 허용합니다. API Key를 `?token=`에 넣으면 거부됩니다.
먼저 [10장](#10-인증-및-토큰-발급)의 Mint Flow로 JWT를 발급받으세요.

### Q: CUDA out of memory
- `CUDA_VISIBLE_DEVICES`가 올바른 GPU를 가리키는지 확인
- 더 작은 양자화(Q4_K_XL)를 사용하거나, `CTX_SIZE`를 줄이세요
- `nvidia-smi`로 다른 프로세스가 VRAM을 점유하고 있지 않은지 확인

### Q: `pip install` 시 guard hook 에러
이 프로젝트의 Claude Code 환경에서는 `pip install`이 guard hook을 트리거합니다.
일반 사용 시에는 이 문제가 없습니다. 정상적으로 `pip install`을 실행하세요.

---

## 18. 아키텍처 개요

```
┌──────────────────────────────────────────────────────────┐
│                     Web UI (React)                       │
│                    localhost:5173                         │
└──────────────────────┬───────────────────────────────────┘
                       │ HTTP + WebSocket (CORS-gated)
                       ▼
┌──────────────────────────────────────────────────────────┐
│              API Gateway (FastAPI/uvicorn)                │
│                    localhost:8080                         │
│  ┌─────────┐ ┌──────────┐ ┌───────────┐ ┌───────────┐  │
│  │  Auth    │ │  CORS    │ │  Trusted  │ │  Security │  │
│  │Middleware│ │Middleware│ │   Host    │ │  Headers  │  │
│  └─────────┘ └──────────┘ └───────────┘ └───────────┘  │
├──────────────────────────────────────────────────────────┤
│                    Application                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Module Registry                     │    │
│  │  (health gating · fault isolation · lifecycle)   │    │
│  └─────────────────────────────────────────────────┘    │
│  ┌──────────┐ ┌────────┐ ┌─────────┐ ┌───────────┐    │
│  │  Model   │ │ DocQA  │ │ Browser │ │  Storage  │    │
│  │ Manager  │ │        │ │         │ │  (SSH)    │    │
│  ├──────────┤ ├────────┤ ├─────────┤ ├───────────┤    │
│  │   LLM    │ │  Exec  │ │   MCP   │ │Multimodal │    │
│  │ Serving  │ │(Docker)│ │(External│ │ (Vision)  │    │
│  │          │ │        │ │ Tools)  │ │           │    │
│  └──────────┘ └────────┘ └─────────┘ └───────────┘    │
│  ┌─────────────────────────────────────────────────┐    │
│  │            Agent Runtime (HITL)                  │    │
│  │  Safety Gate → Approval → Dispatcher → Audit    │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────┬───────────────────────────────────┘
                       │ OpenAI-compatible API
                       ▼
┌──────────────────────────────────────────────────────────┐
│              llama-server (llama.cpp)                     │
│                    localhost:8000                         │
│              Gemma-4 31B (GGUF, GPU×2)                   │
└──────────────────────────────────────────────────────────┘
```

### 핵심 설계 원칙

- **기능 기본 OFF**: 모든 기능은 명시적으로 활성화해야 동작합니다
- **게이트 기반 안전**: 에이전트의 모든 도구 호출은 Safety Gate → Approval → Dispatch 체인을 거칩니다
- **장애 격리**: 하나의 모듈이 실패해도 다른 모듈에 영향을 주지 않습니다
- **비밀 참조 방식**: 비밀번호/토큰은 값이 아닌 환경 변수 이름으로 참조합니다
- **감사 추적**: 에이전트의 모든 게이트 결정과 도구 사용이 기록됩니다

---

## 빠른 시작 요약

최소한의 설정으로 서버를 시작하는 순서:

```bash
# 1. 환경 준비
source ~/anaconda3/etc/profile.d/conda.sh && conda activate local-ai-agent-env-1

# 2. 프로젝트 설치
cd /path/to/local_ai_agent
pip install -e ".[dev]"

# 3. .env 설정 (위 6장 참조 — 최소한 모델 경로 + 비밀 키)

# 4. 서버 시작
local-ai-agent

# 5. 헬스 확인
curl http://127.0.0.1:8080/health

# 6. (선택) Web UI
npm --prefix web install
npm --prefix web run dev
# → http://localhost:5173 에서 접속
```
