import logging

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from .classfier import PictureClassifierPipeline, PictureClassifierPipelineOptions
from hierarchical.postprocessor import ResultPostprocessor
from .chunking import get_chunker, get_pages
import grpc
from ..protos import bge_embed_pb2_grpc
from ..protos import bge_embed_pb2
# from ..retrieval.wnp_opensearch import WnpOpensearch


def get_embed_client(server_address='localhost:50055'):
    """gRPC 임베딩 서비스 클라이언트 생성"""
    channel = grpc.insecure_channel(server_address)
    return bge_embed_pb2_grpc.BgeEmbedStub(channel)

def embed_via_grpc(sentences: list[str], client=None) -> tuple[list[float], dict]:
    if client is None:
        client = get_embed_client()
    
    request = bge_embed_pb2.EmbedRequest(texts=sentences)
    response = client.Embed(request)
    
    if len(response.vectors) == 0:
        return None, None
    
    vector_data = response.vectors[0]
    dense = list(vector_data.dense)
    
    sparse = {str(k): float(v) for k, v in vector_data.sparse.items()}
    return dense, sparse

def test_code():
    client = get_embed_client()
    dense, sparse = embed_via_grpc(["Hello, world!"], client)
    print(dense)
    print(sparse)


# def main():
#     logging.basicConfig(level=logging.INFO)
#     pipeline_options = PictureClassifierPipelineOptions()
#     pipeline_options.images_scale = 2.0
#     pipeline_options.generate_picture_images = True
#     pipeline_options.do_table_structure = True
#     pipeline_options.table_structure_options.do_cell_matching = True
#     pipeline_options.do_ocr = True
#     pipeline_options.generate_table_images = True
#     doc_converter = DocumentConverter(
#         allowed_formats=[
#             InputFormat.PDF,
#             InputFormat.DOCX,
#             InputFormat.HTML,
#             InputFormat.MD,
#             InputFormat.ASCIIDOC,
#             InputFormat.IMAGE
#         ],
#         format_options={
#             InputFormat.PDF: PdfFormatOption(
#                 pipeline_cls=PictureClassifierPipeline,
#                 pipeline_options=pipeline_options,
#             )
#         }
#     )
#     # TXT 파일을 MD로 처리
#     file_path = "/home/kyh/docling/complex_doc.pdf"
    
#     # 확장자만 .md로 변경해서 임시 파일 생성
#     if file_path.endswith('.txt'):
#         import shutil
#         md_file = file_path.replace('.txt', '.md')
#         shutil.copy(file_path, md_file)
#         print(f"📝 TXT -> MD 변환: {file_path} → {md_file}")
#         file_path = md_file
    
#     result = doc_converter.convert(
#         file_path,
#         page_range=(1,1)
#     )

#     # ResultPostprocessor는 PDF 전용 - MD/TXT는 건너뛰기
#     if file_path.endswith('.pdf'):
#         ResultPostprocessor(result).process()
#     chunker = get_chunker(WnpEmbedModel.getModel())
#     chunk_iter = chunker.chunk(dl_doc=result.document)

#     for i, chunk in enumerate(chunk_iter):
#         headings = "No Title" if chunk.meta.headings is None else " > ".join(chunk.meta.headings)
#         content_prefix = f"[{chunk.meta.origin.filename}, {headings}]"
#         content = f"{content_prefix} {chunk.text}"
#         dense, sparse = WnpEmbedModel.embed([content])
#         if(dense is None or sparse is None):
#             print("임베딩 실패")
#             continue
#         pages = list(get_pages(chunk.meta))
#         print(chunk.text)
#         data = {
#             "content": content,
#             "content_dense": dense,    # content_vector.dense 아님!
#             "content_sparse": sparse,
#             "meta": {
#                 "headings": chunk.meta.headings,
#                 "pages": pages,
#             }
#         }
        # WnpOpensearch.put_rag_data(data)
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
    test_code()