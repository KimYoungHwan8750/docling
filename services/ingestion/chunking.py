from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.serializer.markdown import MarkdownTableSerializer
from docling_core.types.doc.document import DocItem

class MDTableSerializerProvider(ChunkingSerializerProvider):
    def get_serializer(self, doc):
        return ChunkingDocSerializer(
            doc=doc,
            table_serializer=MarkdownTableSerializer(),
        )

def get_chunker(embed_model):
    chunker = HybridChunker(
        max_tokens=1024,
        tokenizer = embed_model.tokenizer,
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


