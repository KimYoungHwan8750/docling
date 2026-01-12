from opensearchpy import OpenSearch
from typing import Dict, Any


class OpensearchService:
    """OpenSearch 서비스"""
    
    def __init__(self, client: OpenSearch):
        self.client = client
    
    def search(self, index: str, query: Dict[str, Any], search_pipeline: str = None) -> Dict[str, Any]:
        """OpenSearch에서 검색합니다."""
        params = {}
        if search_pipeline:
            params["search_pipeline"] = search_pipeline
        
        response = self.client.search(
            index=index,
            body=query,
            params=params
        )
        return response
    
    def index_document(self, index: str, document: Dict[str, Any]) -> Dict[str, Any]:
        """문서를 인덱싱합니다."""
        response = self.client.index(
            index=index,
            body=document,
            refresh=True
        )
        return response
    
    def get_client(self) -> OpenSearch:
        """OpenSearch 클라이언트를 반환합니다."""
        return self.client
