from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class EmbedRequest(BaseModel):
    sentences: List[str] = Field(..., description="임베딩할 문장 리스트")


class EmbedResponse(BaseModel):
    dense_vecs: List[List[float]]
    sparse_vecs: List[Dict[str, float]]


class RerankRequest(BaseModel):
    query: str = Field(..., description="검색 쿼리")
    documents: List[str] = Field(..., description="리랭킹할 문서 리스트")


class RerankResponse(BaseModel):
    scores: List[float]


class RAGRequest(BaseModel):
    query: str = Field(..., description="질문")


class RAGResponse(BaseModel):
    answer: str
    context_count: int = Field(default=0, description="사용된 컨텍스트 문서 수")


class AskRequest(BaseModel):
    question: str = Field(..., description="질문")
    user_id: str = Field(..., description="사용자 ID")
    chat_id: str = Field(..., description="채팅 ID")
    chat_seq: int = Field(..., description="채팅 시퀀스")


class AskResponse(BaseModel):
    chat_id: str
    chat_seq: int


class ConvertRequest(BaseModel):
    file_path: str = Field(..., description="변환할 PDF 파일 경로")
    generate_images: bool = Field(default=False, description="이미지 생성 여부")
    images_scale: float = Field(default=2.0, description="이미지 스케일")


class ConvertResponse(BaseModel):
    success: bool
    message: str
    document: Optional[Dict[str, Any]] = None


class ChunkRequest(BaseModel):
    file_path: str = Field(..., description="청킹할 PDF 파일 경로")
    max_tokens: int = Field(default=1024, description="청크당 최대 토큰 수")


class ChunkResponse(BaseModel):
    chunks: List[Dict[str, Any]]


class SearchRequest(BaseModel):
    query: str = Field(..., description="검색 쿼리")
    size: int = Field(default=10, description="검색 결과 수")


class SearchResponse(BaseModel):
    hits: List[Dict[str, Any]]
    total: int


class IndexRequest(BaseModel):
    data: Dict[str, Any] = Field(..., description="색인할 데이터")


class IndexResponse(BaseModel):
    success: bool
    result: Dict[str, Any]
