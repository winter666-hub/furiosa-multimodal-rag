# Furiosa Insight

# 프로젝트 목표

`Furiosa Agentic PDF RAG` 연구 데모용 웹 프론트엔드를 만들어줘.

백엔드는 이미 Render에 배포되어 있다.

## Backend API

Base URL:

```text

https://furiosa-multimodal-rag.onrender.com

```

질문 API:

```http

POST /ask

Content-Type: application/json

```

Request:

```json

{

  "question": "왜 multi-head attention을 사용하는가?"

}

```

Response 예시:

```json

{

  "question": "왜 multi-head attention을 사용하는가?",

  "answer": "...",

  "route": "TEXT_ONLY",

  "routing_reason": "adaptive LLM fallback: LLM classified question as TEXT_ONLY",

  "vision_used": false,

  "vision_available": false,

  "fallback_used": false,

  "sources": [

    {

      "page": 5,

      "chunk": "page-5-chunk-1"

    },

    {

      "page": 4,

      "chunk": "page-4-chunk-1"

    }

  ],

  "latency_ms": {

    "total": 15126.6,

    "routing": 524.5

  }

}

```

# UI 목표

연구 프로젝트 데모처럼 깔끔하고 전문적인 AI 챗봇 UI를 만들어줘.

현재 문서는:

```text

Attention Is All You Need

```

한 편으로 고정되어 있다.

## 페이지 구성

상단:

```text

Furiosa Agentic PDF RAG

Selective Multimodal RAG Demo

```

간단한 설명:

```text

Ask questions about "Attention Is All You Need".

The system adaptively decides whether visual reasoning is required.

```

중앙에는 대화형 chatbot UI를 배치한다.

사용자가 질문을 입력하고 Send 버튼을 누르면

Render의 `/ask` API를 호출한다.

# Chat UI

사용자 질문과 AI 답변을 chat bubble 형태로 표시한다.

AI 답변은 Markdown을 렌더링한다.

가능하면 다음도 렌더링한다.

- Markdown

- bullet list

- inline code

- LaTeX / mathematical expressions

예:

```text

MultiHead(Q,K,V) = Concat(head_1, ..., head_h)W^O

```

# Answer metadata

각 AI 응답 하단에 작은 metadata 영역을 만든다.

표시:

- Route

- Vision Used

- Fallback Used

- Total Latency

- Sources

예:

```text

Route: TEXT_ONLY

Vision: Not used

Latency: 15.1s

Sources: Page 5 · Page 4 · Page 3

```

Route는 badge 형태로 표시한다.

예:

```text

TEXT_ONLY

VISUAL_REQUIRED

```

# Visual fallback 표시

백엔드가 다음과 같이 반환하면:

```json

{

  "route": "VISUAL_REQUIRED",

  "vision_used": false,

  "vision_available": false,

  "fallback_used": true

}

```

사용자에게 오류처럼 보이지 않도록 작은 안내를 표시한다.

예:

```text

Visual reasoning was requested, but the web demo is currently running in hosted-only mode.

The answer was generated using text RAG fallback.

```

# Loading 상태

백엔드 응답이 수 초에서 수십 초 걸릴 수 있다.

Send 후 반드시 loading 상태를 표시한다.

예:

```text

Analyzing the paper...

```

또는 animated typing indicator를 사용한다.

중복 요청을 막기 위해 응답 대기 중에는 Send 버튼을 disable한다.

# Error handling

다음 오류를 사용자 친화적으로 처리한다.

- Network error

- HTTP 422

- HTTP 502

- HTTP 503

- HTTP 500

- Render cold start

예:

```text

The AI service is waking up or temporarily unavailable. Please try again.

```

내부 traceback이나 API 세부 오류를 노출하지 않는다.

# Suggested questions

첫 화면에 클릭 가능한 예시 질문을 제공한다.

```text

왜 multi-head attention을 사용하는가?

scaled dot-product attention에서 sqrt(d_k)로 나누는 이유는?

Figure 1에서 Encoder와 Decoder 구조의 차이는?

Encoder의 출력은 Decoder의 어느 attention block으로 연결되는가?

```

클릭하면 입력창에 채워지거나 즉시 질문할 수 있게 한다.

# 디자인

스타일:

- clean

- modern

- academic / AI research demo

- 과도한 장식 금지

- desktop과 mobile 모두 responsive

연구 데모라는 점이 드러나게 하되

대시보드처럼 복잡하게 만들지 않는다.

# 중요한 제한사항

- Furiosa API key를 frontend에 넣지 마라.

- Render API만 frontend에서 호출한다.

- Backend 로직을 frontend에 재구현하지 마라.

- Router 판단을 frontend에서 하지 마라.

- fake answer를 만들지 마라.

- 실제 `/ask` 응답을 그대로 사용한다.

# 완료 목표

사용자가 브라우저에서:

```text

질문 입력

→ Send

→ Render /ask 호출

→ loading

→ AI 답변 표시

→ route / source / latency 확인

```

까지 수행할 수 있는 완성된 데모 페이지를 만들어줘.

This project was built with [Lovable](https://lovable.dev).

**Live app**: https://furiosa-insight-ask.lovable.app

## Build with Lovable

Continue developing this project in the [Lovable editor](https://lovable.dev/projects/65fc7b6c-0197-4401-b4aa-626489e75e2e).

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: every change made in Lovable is committed straight to this repository.
- **Full ownership**: this code is yours. Push to `main` on GitHub and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```
