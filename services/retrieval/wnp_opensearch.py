from contextlib import asynccontextmanager
from typing import Any
from opensearchpy import OpenSearch
from pydantic import BaseModel, Field
from wnp_reranker import WnpReranker
from fastapi import FastAPI
import httpx
app = FastAPI()

class GetDocsModel(BaseModel):
    query: str = Field(..., description="검색 쿼리")

class PutDocsModel(BaseModel):
    content: str = Field(..., description="문서 내용")
    content_dense: list[float] = Field(..., description="문서 임베딩 벡터")
    content_sparse: dict[str, float] = Field(..., description="문서 임베딩 희소 벡터")
    meta: dict[str, Any] = Field(..., description="문서 메타데이터")

app_service = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    _host = 'localhost'
    _port = 9200
    _client = OpenSearch(
        hosts = [{'host': _host, 'port': _port}],
    )
    app_service['client'] = _client
    yield
    app_service['client'].close()
    app_service.clear()

@app.post("/docs")
def get_docs(request: GetDocsModel):
    search_query = {
        "size": 10,
        "query": {
            "hybrid": {
                "queries": [
                    # {"match": {"content": query}},
                    {"knn": {"content_dense": {"vector": dense_vec, "k": 10}}},
                    {"bool": {"should": sparse_should_clauses}}
                ]
            }
        }
    }


class WnpOpensearch:
    _host = 'localhost'
    _port = 9200
    _client = OpenSearch(
        hosts = [{'host': _host, 'port': _port}],
    )

    def get_client() -> OpenSearch:
        return WnpOpensearch._client

    def put_rag_data(data):
        response = WnpOpensearch.get_client().index(
            index="rag_data",
            body=data,
            refresh=True
        )
        return response