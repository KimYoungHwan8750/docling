from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc.document import DocItem
from typing import List, Dict, Any


class MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )


class ChunkingService:
    """문서 청킹 서비스"""
    
    def __init__(self, embed_model):
        self.embed_model = embed_model
    
    def get_chunker(self, max_tokens: int = 1024):
        """청커를 생성합니다."""
        chunker = HybridChunker(
            max_tokens=max_tokens,
            tokenizer=self.embed_model.tokenizer,
            serializer_provider=MDTableSerializerProvider()
        )
        return chunker
    
    def chunk_document(self, document, max_tokens: int = 1024) -> List[Dict[str, Any]]:
        """문서를 청킹합니다."""
        chunker = self.get_chunker(max_tokens)
        chunk_iter = chunker.chunk(document)
        
        chunks = []
        for chunk in chunk_iter:
            pages = self.get_pages(chunk.meta)
            chunks.append({
                "text": chunk.text,
                "pages": sorted(list(pages)),
                "meta": str(chunk.meta)
            })
        
        return chunks
    
    @staticmethod
    def get_pages(meta):
        """청크의 페이지 번호들을 추출합니다."""
        pages = set()
        for item in meta.doc_items:
            if isinstance(item, DocItem):
                for prov in item.prov:
                    pages.add(prov.page_no)
        return pages
