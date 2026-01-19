import logging
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from FlagEmbedding import BGEM3FlagModel, FlagReranker
from opensearchpy import OpenSearch
from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption

from .models import (
    EmbedRequest, EmbedResponse,
    RerankRequest, RerankResponse,
    RAGRequest, RAGResponse,
    ConvertRequest, ConvertResponse,
    ChunkRequest, ChunkResponse,
    SearchRequest, SearchResponse,
    IndexRequest, IndexResponse,
    AskRequest, AskResponse
)
from .services.embed_service import EmbedService
from .services.reranker_service import RerankerService
from .services.opensearch_service import OpensearchService
from .services.chunking_service import ChunkingService
from .services.classifier_service import ClassifierService, PictureClassifierPipelineOptions
from .services.rag_service import RAGService
from fastapi.middleware.cors import CORSMiddleware
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 전역 서비스 객체
app_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행되는 lifespan 이벤트"""
    logger.info("🚀 서비스 초기화 중...")
    
    # 1. 임베딩 모델 로드
    logger.info("📦 임베딩 모델 로딩 중... (BAAI/bge-m3)")
    embed_model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    app_state['embed_model'] = embed_model
    app_state['embed_service'] = EmbedService(embed_model)
    logger.info("✅ 임베딩 모델 로드 완료")
    
    # 2. 리랭커 모델 로드
    logger.info("📦 리랭커 모델 로딩 중... (BAAI/bge-reranker-v2-m3)")
    reranker_model = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
    app_state['reranker_model'] = reranker_model
    app_state['reranker_service'] = RerankerService(reranker_model)
    logger.info("✅ 리랭커 모델 로드 완료")
    
    # 3. OpenSearch 클라이언트
    logger.info("📦 OpenSearch 클라이언트 초기화 중...")
    opensearch_client = OpenSearch(
        hosts=[{'host': 'localhost', 'port': 9200}],
    )
    app_state['opensearch_client'] = opensearch_client
    app_state['opensearch_service'] = OpensearchService(opensearch_client)
    logger.info("✅ OpenSearch 클라이언트 초기화 완료")
    
    # 4. Docling DocumentConverter
    logger.info("📦 Docling DocumentConverter 초기화 중...")
    pipeline_options = PictureClassifierPipelineOptions()
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
    app_state['doc_converter'] = doc_converter
    app_state['classifier_service'] = ClassifierService(doc_converter)
    logger.info("✅ Docling DocumentConverter 초기화 완료")
    
    # 5. 청킹 서비스
    app_state['chunking_service'] = ChunkingService(embed_model)
    logger.info("✅ 청킹 서비스 초기화 완료")
    
    # 6. RAG 서비스
    vllm_url = "http://localhost:8000/v1/chat/completions"
    app_state['rag_service'] = RAGService(
        app_state['embed_service'],
        app_state['reranker_service'],
        app_state['opensearch_service'],
        vllm_url
    )
    logger.info("✅ RAG 서비스 초기화 완료")
    
    logger.info("🎉 모든 서비스 초기화 완료!")
    
    yield
    
    # 종료 시 정리
    logger.info("🛑 서비스 종료 중...")
    app_state.clear()
    logger.info("✅ 서비스 종료 완료")


app = FastAPI(
    title="Docling RAG API",
    description="문서 처리 및 RAG 기반 질의응답 API",
    version="1.0.0",
    lifespan=lifespan
)

origins = [
    "*",                           # 모든 곳에서 접속을 허용하려면 "*" 사용 (테스트용)
]

# 3. 미들웨어 등록
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # 실 운영 시에는 origins 목록으로 제한하는 것이 안전합니다.
    allow_credentials=True,
    allow_methods=["*"],           # 모든 HTTP 메서드(GET, POST 등) 허용
    allow_headers=["*"],           # 모든 HTTP 헤더 허용
)


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Docling RAG API",
        "version": "1.0.0",
        "endpoints": [
            "/embed",
            "/rerank",
            "/rag",
            "/api/ask",
            "/api/stream",
            "/convert",
            "/chunk",
            "/search",
            "/index"
        ]
    }


# 세션 저장소 (간단한 in-memory 저장소)
sessions = {}


@app.post("/api/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """질문을 받아서 세션에 저장하고 chat_id와 chat_seq를 반환합니다."""
    try:
        # 세션에 질문 저장
        session_key = f"{request.user_id}___{request.chat_id}"
        
        if session_key not in sessions:
            sessions[session_key] = []
        
        sessions[session_key].append({
            "role": "user",
            "content": request.question,
            "seq": request.chat_seq
        })
        
        return AskResponse(
            chat_id=request.chat_id,
            chat_seq=request.chat_seq
        )
    except Exception as e:
        logger.error(f"질문 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stream")
async def stream(
    user_id: str = Query(..., description="사용자 ID"),
    chat_id: str = Query(..., description="채팅 ID"),
    chat_seq: int = Query(..., description="채팅 시퀀스")
):
    """SSE를 통해 스트리밍 응답을 반환합니다."""
    try:
        # 세션에서 질문 가져오기
        session_key = f"{user_id}___{chat_id}"
        
        if session_key not in sessions or not sessions[session_key]:
            raise HTTPException(status_code=404, detail="세션을 찾을 수 없습니다.")
        
        # 마지막 질문 가져오기
        last_message = sessions[session_key][-1]
        query = last_message.get("content", "")
        
        if not query:
            raise HTTPException(status_code=400, detail="질문이 비어있습니다.")
        
        # RAG 서비스로 스트리밍 생성
        rag_service = app_state['rag_service']
        
        async def event_generator():
            try:
                async for event in rag_service.generate_answer_stream(query):
                    yield event
            except Exception as e:
                logger.error(f"스트리밍 생성 중 오류: {e}")
                yield f"event: error\ndata: {json.dumps({'msg': str(e)})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "Transfer-Encoding": "chunked"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"스트리밍 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest):
    """텍스트를 임베딩합니다."""
    try:
        embed_service = app_state['embed_service']
        dense_vecs, sparse_vecs = embed_service.embed(request.sentences)
        return EmbedResponse(dense_vecs=dense_vecs, sparse_vecs=sparse_vecs)
    except Exception as e:
        logger.error(f"임베딩 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """문서들을 리랭킹합니다."""
    try:
        reranker_service = app_state['reranker_service']
        scores = reranker_service.rerank(request.query, request.documents)
        return RerankResponse(scores=scores)
    except Exception as e:
        logger.error(f"리랭킹 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rag", response_model=RAGResponse)
async def rag(request: RAGRequest):
    """RAG 기반 질의응답을 수행합니다."""
    print(request.query)
    try:
        rag_service = app_state['rag_service']
        answer, context_count = rag_service.generate_answer(request.query)
        return RAGResponse(answer=answer, context_count=context_count)
    except Exception as e:
        logger.error(f"RAG 처리 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/convert", response_model=ConvertResponse)
async def convert(request: ConvertRequest):
    """PDF 문서를 변환합니다."""
    try:
        classifier_service = app_state['classifier_service']
        document_dict = classifier_service.convert_document(request.file_path)
        return ConvertResponse(
            success=True,
            message="문서 변환 완료",
            document=document_dict
        )
    except Exception as e:
        logger.error(f"문서 변환 중 오류 발생: {e}")
        return ConvertResponse(
            success=False,
            message=f"문서 변환 실패: {str(e)}",
            document=None
        )


@app.post("/chunk", response_model=ChunkResponse)
async def chunk(request: ChunkRequest):
    """문서를 청킹합니다."""
    try:
        # 먼저 문서 변환
        classifier_service = app_state['classifier_service']
        result = app_state['doc_converter'].convert(request.file_path)
        
        # 청킹
        chunking_service = app_state['chunking_service']
        chunks = chunking_service.chunk_document(result.document, request.max_tokens)
        
        return ChunkResponse(chunks=chunks)
    except Exception as e:
        logger.error(f"청킹 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """OpenSearch에서 검색합니다."""
    try:
        # 쿼리 임베딩
        embed_service = app_state['embed_service']
        dense_vec, sparse_vec = embed_service.embed_single(request.query)
        
        # 희소 벡터 필터링
        threshold = 0.1
        max_top_k = 30
        filtered_sparse = {t: w for t, w in sparse_vec.items() if w >= threshold}
        
        if len(filtered_sparse) > max_top_k:
            top_sparse = dict(sorted(filtered_sparse.items(), key=lambda x: x[1], reverse=True)[:max_top_k])
        else:
            top_sparse = filtered_sparse
        
        # 검색 쿼리 구성
        sparse_should_clauses = []
        for token, weight in top_sparse.items():
            sparse_should_clauses.append({
                "rank_feature": {
                    "field": f"content_sparse.{token}",
                    "boost": float(weight)
                }
            })
        
        search_query = {
            "size": request.size,
            "query": {
                "hybrid": {
                    "queries": [
                        {"knn": {"content_dense": {"vector": dense_vec, "k": request.size}}},
                        {"bool": {"should": sparse_should_clauses}}
                    ]
                }
            }
        }
        
        # 검색 실행
        opensearch_service = app_state['opensearch_service']
        result = opensearch_service.search(
            index="rag_data",
            query=search_query,
            search_pipeline="hybrid_search_pipeline2"
        )
        
        hits = result['hits']['hits']
        total = result['hits']['total']['value']
        
        return SearchResponse(hits=hits, total=total)
    except Exception as e:
        logger.error(f"검색 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", response_model=IndexResponse)
async def index_document(request: IndexRequest):
    """문서를 OpenSearch에 인덱싱합니다."""
    try:
        opensearch_service = app_state['opensearch_service']
        result = opensearch_service.index_document("rag_data", request.data)
        return IndexResponse(success=True, result=result)
    except Exception as e:
        logger.error(f"인덱싱 중 오류 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {
        "status": "healthy",
        "services": {
            "embed_model": "embed_model" in app_state,
            "reranker_model": "reranker_model" in app_state,
            "opensearch": "opensearch_client" in app_state,
            "doc_converter": "doc_converter" in app_state,
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
