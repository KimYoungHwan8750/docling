from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc.document import DocItem
from transformers import AutoTokenizer

class MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )

def get_chunker(tokenizer=None):
    """
    Chunker 생성. tokenizer가 없으면 기본 BGE-M3 tokenizer 로드
    임베딩 서버와 분리되어 있어도 tokenizer만 로컬에 로드하면 됨
    """
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    
    chunker = HybridChunker(
        max_tokens=1024,
        tokenizer=tokenizer,
        serializer_provider=MDTableSerializerProvider()
    )
    return chunker

# 청킹된 데이터가 여러 페이지에 걸쳐 있을 수 있기 때문
def get_pages(meta):
    pages = set()
    for item in meta.doc_items:
        if isinstance(item, DocItem):
            for prov in item.prov:
                pages.add(prov.page_no)
    return pages    


