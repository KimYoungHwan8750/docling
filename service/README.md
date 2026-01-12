# Docling RAG API Service

FastAPI 기반의 문서 처리 및 RAG(Retrieval-Augmented Generation) 질의응답 서비스입니다.

## 주요 기능

- **임베딩**: BGE-M3 모델을 사용한 문서/쿼리 임베딩 (Dense + Sparse)
- **리랭킹**: BGE-Reranker-v2-m3를 사용한 검색 결과 리랭킹
- **문서 변환**: Docling을 사용한 PDF 문서 파싱 및 변환
- **문서 청킹**: 하이브리드 청킹을 통한 문서 분할
- **검색**: OpenSearch 기반 하이브리드 검색 (Dense + Sparse)
- **RAG 질의응답**: 검색된 문서를 기반으로 한 답변 생성

## 설치

```bash
cd service
pip install -r requirements.txt
```

## 실행

```bash
# 개발 서버 실행
python main.py

# 또는 uvicorn으로 실행
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

## API 엔드포인트

### 1. 임베딩
```http
POST /embed
Content-Type: application/json

{
  "sentences": ["문장1", "문장2"]
}
```

### 2. 리랭킹
```http
POST /rerank
Content-Type: application/json

{
  "query": "검색 쿼리",
  "documents": ["문서1", "문서2", "문서3"]
}
```

### 3. RAG 질의응답
```http
POST /rag
Content-Type: application/json

{
  "query": "질문"
}
```

### 4. 문서 변환
```http
POST /convert
Content-Type: application/json

{
  "file_path": "/path/to/document.pdf",
  "generate_images": false,
  "images_scale": 2.0
}
```

### 5. 문서 청킹
```http
POST /chunk
Content-Type: application/json

{
  "file_path": "/path/to/document.pdf",
  "max_tokens": 1024
}
```

### 6. 검색
```http
POST /search
Content-Type: application/json

{
  "query": "검색 쿼리",
  "size": 10
}
```

### 7. 인덱싱
```http
POST /index
Content-Type: application/json

{
  "data": {
    "content": "문서 내용",
    "content_dense": [...],
    "content_sparse": {...}
  }
}
```

### 8. 헬스 체크
```http
GET /health
```

## 아키텍처

### Lifespan 관리
- 애플리케이션 시작 시 모든 모델과 서비스를 메모리에 로드
- 임베딩 모델, 리랭커 모델, Docling 컨버터가 한 번만 초기화됨
- 빠른 응답 속도와 효율적인 리소스 사용

### 서비스 구조
```
service/
├── main.py                      # FastAPI 앱 및 엔드포인트
├── models.py                    # Pydantic 모델들
├── requirements.txt             # 의존성
├── README.md                    # 문서
└── services/
    ├── __init__.py
    ├── embed_service.py         # 임베딩 서비스
    ├── reranker_service.py      # 리랭커 서비스
    ├── opensearch_service.py    # OpenSearch 서비스
    ├── chunking_service.py      # 청킹 서비스
    ├── classifier_service.py    # 문서 분류/변환 서비스
    └── rag_service.py           # RAG 서비스
```

## 환경 요구사항

- Python 3.9+
- OpenSearch 실행 중 (localhost:9200)
- vLLM 서버 실행 중 (localhost:8000)
- GPU 권장 (fp16 사용)

## 참고사항

- 모든 모델은 애플리케이션 시작 시 메모리에 로드되므로 초기 로딩 시간이 소요됩니다.
- GPU 메모리가 충분한지 확인하세요 (권장: 16GB 이상).
- OpenSearch와 vLLM 서버가 실행 중이어야 합니다.
