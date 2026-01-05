import logging

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import PictureItem, TableItem, TextItem
from Classfier import PictureClassifierPipeline, PictureClassifierPipelineOptions
from docling.chunking import HybridChunker
from hierarchical.postprocessor import ResultPostprocessor
from FlagEmbedding import BGEM3FlagModel
from chunking import get_chunker, get_pages
from embed import WnpEmbedModel
from WnpOpensearch import WnpOpensearch

def main():
    logging.basicConfig(level=logging.INFO)

    pipeline_options = PictureClassifierPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.do_ocr = True
    pipeline_options.generate_table_images = True
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=PictureClassifierPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )
    result = doc_converter.convert(
        "/home/kyh/docling/complex_doc.pdf",
        # page_range=(4, 5)
    )

    ResultPostprocessor(result).process()
    chunker = get_chunker()
    chunk_iter = chunker.chunk(dl_doc=result.document)

    for i, chunk in enumerate(chunk_iter):
        headings = "No Title" if chunk.meta.headings is None else " > ".join(chunk.meta.headings)
        content_prefix = f"[{chunk.meta.origin.filename}, {headings}]"
        content = f"{content_prefix} {chunk.text}"
        dense, sparse = WnpEmbedModel.embed([content])
        if(dense is None or sparse is None):
            print("임베딩 실패")
            continue
        pages = list(get_pages(chunk.meta))
        data = {
            "content": content,
            "content_dense": dense,    # content_vector.dense 아님!
            "content_sparse": sparse,
            "meta": {
                "headings": chunk.meta.headings,
                "pages": pages,
            }
        }
        WnpOpensearch.put_rag_data(data)
        # if hasattr(chunk, 'headings')
        # print(f"Chunk {i}: {content}")    
    # for i, (item, level) in enumerate(result.document.iterate_items()):
    #     if isinstance(item, TableItem):
    #         print("="*30)
    #         print(f"Table {i}: {item.export_to_markdown()}")
    #     else:
    #         print(f"Item {i}: {type(item).__name__}")

    # for i, (item, level) in enumerate(result.document.iterate_items()):
        
    #     # [A] 표(Table)나 그림(Picture)은 청킹 없이 통째로 처리
    #     if isinstance(item, (TableItem, PictureItem)):
    #         # print(f"--- [Atomic Item] Index {i} ({type(item).__name__}) ---")
            
    #         # 표라면 마크다운, 그림이라면 캡션/설명을 추출
    #         # content = item.export_to_markdown() if isinstance(item, TableItem) else item.text
            
    #         # 이 데이터를 바로 임베딩 모델에 넣거나 DB에 저장 준비
    #         # print(content) 

    #     # [B] 일반 텍스트 아이템들만 청커에게 전달
    #         pass
    #     elif isinstance(item, TextItem):
    #         # 단일 아이템에 대해서도 계층 구조를 유지하며 청킹 가능
    #         # chunk() 메서드에 단일 아이템을 리스트로 감싸서 보냅니다.
    #         text_chunks = chunker.chunk(dl_doc=result.document, items=[item])
            
    #         for j, chunk in enumerate(text_chunks):
    #             # chunk.text에는 계층 정보가 직렬화되어 포함될 수 있습니다.
    #             print(f"--- [Text Chunk] Index {i}-{j} ---")
    #             print(chunk.text)

if __name__ == "__main__":
    main()