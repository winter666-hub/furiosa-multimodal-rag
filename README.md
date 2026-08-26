# Furiosa NPU Multimodal RAG Agent

연구 논문 PDF를 업로드하고 질문한 뒤, Adaptive routing과 retrieval/reranking을 거쳐
근거 페이지까지 확인할 수 있는 범용 Paper RAG 프로젝트입니다. 기존
`Attention Is All You Need` 문서는 배포 데모의 example document로도 사용할 수 있습니다.

```text
Upload a research paper PDF
→ Ask questions with its document_id
→ Adaptive routing
→ Retrieval + reranking
→ RAG answer
→ Inspect one-based source pages
```

Render의 `hosted_only` web mode에서는 Direct NPU Vision을 호출하지 않으며,
`VISUAL_REQUIRED` 질문도 routing 결과를 보존한 채 해당 문서의 Text RAG로 fallback합니다.

## 현재 구조

```text
app/                         # 향후 UI/API entrypoint
data/                        # 업로드 문서와 로컬 벡터 저장소(버전 관리 제외)
src/furiosa_rag/
  clients/                   # 공통 HTTP/오류 처리
  cli/                       # 운영/개발 명령
  config.py                  # 환경변수 설정
  llm.py                     # LLM 인터페이스와 Furiosa Chat API
  embedding.py               # Embedding 인터페이스와 Furiosa API
  reranker.py                # Reranker 인터페이스와 Furiosa API
  providers/interfaces.py    # OCR/검색/번역/문서 파싱 provider 규약
tests/                       # 네트워크 없이 실행되는 단위 테스트
```

`providers/interfaces.py`에는 향후 외부 OCR, 검색, 번역, 문서 파싱 서비스를 연결하기 위한
`Protocol`과 provider-neutral 결과 타입만 정의되어 있습니다. 현재 외부 API 구현, SDK,
API key 또는 네트워크 호출은 포함하지 않습니다. 새 provider는 해당 인터페이스를 구현하고
파이프라인 조립 단계에서 주입합니다.

## 연결 확인

### 1. SSH 터널 열기 (Git Bash)

NPU 서버가 SSH 키 파일을 통해서만 접근 가능하다면 먼저 로컬 포트 포워딩을 엽니다.
키 파일과 실제 접속 정보는 Git에 커밋하지 않습니다.

```bash
cp .ssh-tunnel.example .ssh-tunnel.env
# .ssh-tunnel.env에서 SSH_KEY, SSH_USER, SSH_HOST와 원격 포트를 수정
bash scripts/start_ssh_tunnel.sh
```

이 터미널은 연결 테스트와 앱 실행 중 계속 열어 둡니다. 스크립트는 기본적으로 NPU 서버의
`8000~8003`을 Windows PC의 `127.0.0.1:8000~8003`으로 전달합니다. 서버에서 네 모델이
다른 포트를 사용하면 `.ssh-tunnel.env`의 `REMOTE_*_PORT`만 변경하면 됩니다.
Vision이 단독으로 원격 8000에서 실행되는 현재 테스트 구성이라면 `ENABLE_LLM=false`,
`ENABLE_EMBEDDING=false`, `ENABLE_RERANKER=false`, `LOCAL_VISION_PORT=8000`,
`REMOTE_VISION_PORT=8000`으로 설정하고 `FURIOSA_VISION_BASE_URL=http://localhost:8000/v1`을
사용할 수 있습니다. 애플리케이션의 endpoint 기준값은 포트가 아니라 `.env` 설정입니다.

직접 명령을 사용할 경우 LLM 한 개에 대한 최소 명령은 다음과 같습니다.

```bash
ssh -i /c/path/to/npu.pem -N -L 8000:127.0.0.1:8000 ubuntu@npu.example.com
```

### 2. Python 연결 진단

1. `.env.example`을 `.env`로 복사하고 실제 서버 주소와 모델 ID를 입력합니다.
2. 개발 모드로 설치하고 진단 명령을 실행합니다.

```powershell
Copy-Item .env.example .env
python -m pip install -e ".[dev]"
furiosa-check
```

설치하지 않고도 실행할 수 있습니다.

```powershell
$env:PYTHONPATH = "src"
python -m furiosa_rag.cli.check_connection
```

`/v1/models` 연결 확인에 성공하면 실제 Text RAG API에 각각 한 번씩 최소 추론 요청을
보냅니다. Embedding → Reranker → LLM 순서이며 Vision API는 아직 호출하지 않습니다.

```powershell
$env:PYTHONPATH = "src"
python -m furiosa_rag.cli.smoke_test
```

Git Bash에서는 다음과 같이 실행할 수 있습니다.

```bash
PYTHONPATH=src python -m furiosa_rag.cli.smoke_test
```

성공 시 `Inference smoke test: 3/3 APIs passed`가 출력됩니다. 실패 시 API URL, HTTP 상태
또는 timeout 원인이 출력되고 종료 코드 `1`을 반환합니다.

명령은 LLM, Vision, Embedding, Reranker 서버 각각의 `GET /v1/models`를 호출해 HTTP
상태, latency, 서버가 보고한 모델 ID를 출력합니다. 하나라도 실패하면 종료 코드 `1`을
반환합니다. `.env`는 외부 패키지 없이 자동으로 읽으며 이미 설정된 환경변수를 덮어쓰지
않습니다.

단일 서버만 먼저 확인하려면 나머지 URL을 비워둘 수 있습니다.

```dotenv
FURIOSA_LLM_BASE_URL=http://npu-server:8000/v1
FURIOSA_VISION_BASE_URL=
FURIOSA_EMBEDDING_BASE_URL=
FURIOSA_RERANKER_BASE_URL=
```

## Public demo deployment safety

The hosted demo applies process-local IP rate limits to PDF uploads and
questions, limits concurrent model work, and rejects PDFs above 25 MB. Uploaded
document directories are bounded by a six-hour TTL, a 20-document count cap,
and a 500 MB aggregate storage cap by default. Cleanup protects documents that
are currently being indexed or queried.

These are minimum safeguards for a single Render demo instance, not distributed
production security. The limiter state resets on restart, and Render's local
filesystem is ephemeral. Multi-instance or higher-volume deployments should use
Cloudflare Rate Limiting, Durable Objects, Redis, or another shared limiter and
durable object storage. See `DEPLOY_RENDER.md` for all configuration variables.

The Cloudflare Worker and Render service must share the same server-side
`PAPER_RAG_PROXY_SECRET`. This allows Render to trust the Worker's
`CF-Connecting-IP` forwarding without trusting spoofable client headers. The
proxy secret is independent of `FURIOSA_API_KEY`; neither value belongs in the
browser bundle or repository.

## 테스트

```powershell
python -m pip install -r requirements.txt
python -m pytest
```

## Document embedding cache

Text RAG 실행 시 PDF hash, chunk 설정, embedding 모델 ID가 같은 캐시를
`data/cache/embeddings/`에서 자동으로 재사용합니다. 강제로 다시 생성하려면
`--rebuild-cache`를 추가합니다.

```powershell
$env:PYTHONPATH = "src"
python -m furiosa_rag.cli.run_rag "data/attention_is_all_you_need.pdf" "Why do the authors use multi-head attention?" --rebuild-cache
```

멀티모달 경로는 reranking 결과의 최상위 페이지 하나만 렌더링하고 Vision에 전달하며,
Vision 호출 실패 시 같은 검색 결과를 사용한 Text RAG로 자동 fallback합니다.
Vision 응답 길이는 `FURIOSA_VISION_MAX_TOKENS`로 조절하며 기본값은 `256`입니다.

```powershell
python -m furiosa_rag.cli.run_multimodal_rag "data/attention_is_all_you_need.pdf" "Figure 1에서 Encoder와 Decoder의 차이는?" --top-k 3 --top-n 3
```

일회성 benchmark에서는 `--vision-max-tokens 128`처럼 환경설정을 덮어쓸 수 있습니다.

Top-k별 latency 비교 결과는 다음 명령으로 CSV에 저장할 수 있습니다.

```powershell
python -m furiosa_rag.cli.benchmark "data/attention_is_all_you_need.pdf" "Why do the authors use multi-head attention?" --top-k 3 5 10 20 --top-n 3 --output "data/benchmarks/attention_top_k.csv"
```

## Router benchmark

The deterministic rule router can be evaluated without Furiosa APIs, vision models, or an NPU:

```powershell
$env:PYTHONPATH = "src"
python -m furiosa_rag.cli.benchmark_router benchmarks/router_eval.jsonl --output benchmarks/router_results.csv
```
