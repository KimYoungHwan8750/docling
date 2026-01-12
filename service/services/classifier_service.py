from docling_core.types.doc import PictureItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionApiOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from typing import Dict, Any, Optional


class PictureClassifierPipelineOptions(PdfPipelineOptions):
    """이미지 분류 파이프라인 옵션"""
    do_picture_classifer: bool = True
    do_ocr: bool = True
    generate_picture_images: bool = False
    enable_remote_services: bool = True
    images_scale: float = 2.0
    do_table_structure: bool = True
    do_picture_description: bool = True
    picture_description_options: PictureDescriptionApiOptions = PictureDescriptionApiOptions(
        url="http://192.168.0.99:8000/v1/chat/completions",
        params=dict(
            model="My_Model",
            seed=42,
            max_completion_tokens=250,
            temperature=0
        ),
        prompt="""
이미지를 '바', '도넛', '이미지' 세 형태로 분류하고 이후 이미지에 대한 내용을 설명해줘. 데이터가 있는 경우 누락되는 데이터 없게 마크다운으로 잘 요약해줘. 자료는 이미지에 있는 언어 그대로 정리해
예: [도넛] 이 이미지는 나이대별 결제 수단에 대해 설명하고 있습니다.
1살~19살:
1. 현금: 24%
2. 수표: 30%
""",
        timeout=300
    )


class ClassifierService:
    """문서 분류/변환 서비스"""
    
    def __init__(self, doc_converter: DocumentConverter):
        self.doc_converter = doc_converter
    
    def convert_document(self, file_path: str) -> Dict[str, Any]:
        """문서를 변환합니다."""
        result = self.doc_converter.convert(file_path)
        
        # 결과 처리
        document_dict = {
            "text": result.document.export_to_markdown(),
            "pictures": []
        }
        
        # 이미지 정보 추출
        for element, _level in result.document.iterate_items():
            if isinstance(element, PictureItem):
                document_dict["pictures"].append({
                    "ref": element.self_ref,
                    "caption": element.caption_text(doc=result.document),
                    "annotations": str(element.annotations)
                })
        
        return document_dict
