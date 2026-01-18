from contextlib import asynccontextmanager
from typing import Any
from opensearchpy import OpenSearch
from pydantic import BaseModel, Field
from fastapi import FastAPI

class PutDocsMetaModel(BaseModel):
    headings: list[str] = Field(..., description="문서 제목")
    pages: list[int] = Field(..., description="문서 페이지")

class GetDocsModel(BaseModel):
    size: int = Field(..., description="검색 결과 개수")
    query: dict[str, Any] = Field(..., description="검색 쿼리")

class PutDocsModel(BaseModel):
    content: str = Field(..., description="문서 내용")
    content_dense: list[float] = Field(..., description="문서 임베딩 벡터")
    content_sparse: dict[str, float] = Field(..., description="문서 임베딩 희소 벡터")
    meta: dict[str, PutDocsMetaModel] = Field(..., description="문서 메타데이터")

@asynccontextmanager
async def lifespan(app: FastAPI):
    _host = 'localhost'
    _port = 9200
    app.state.client = OpenSearch(
        hosts = [{'host': _host, 'port': _port}],
    )
    yield
    app.state.client.close()

app = FastAPI(lifespan=lifespan)

@app.post("/docs")
def get_docs(request: GetDocsModel):
    response = app.state.client.search(
        index="rag_data",
        body=request.model_dump(),
        params={"search_pipeline": "hybrid_search_pipeline2"}
    )
    print(response)
    return response['hits']['hits']

@app.put("/docs")
def put_docs(request: PutDocsModel):
    data = {
        "content": request.content,
        "content_dense": request.content_dense,
        "content_sparse": request.content_sparse,
        "meta": request.meta
    }
    app.state.client.index(
        index="rag_data",
        body=data,
        refresh=True
    )
    return {"message": "문서 저장 완료"}