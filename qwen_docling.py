from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import VlmPipelineOptions
# 상세 설정을 위한 모듈 추가
from docling.datamodel.pipeline_options_vlm_model import (
    InlineVlmOptions, 
    InferenceFramework, 
    TransformersModelType,
    ResponseFormat # ✅ ResponseFormat을 가져옵니다.
)

# ✅ 수정된 부분: prompt와 response_format 인자 추가
pipeline_options = VlmPipelineOptions(
    vlm_options=InlineVlmOptions(
        repo_id="Qwen/Qwen2.5-VL-7B-Instruct",
        prompt="Convert this page to markdown. Do not miss any text and only output the bare markdown!", # ✅ 추가
        response_format=ResponseFormat.MARKDOWN, # ✅ 추가
        inference_framework=InferenceFramework.TRANSFORMERS,
        transformers_model_type=TransformersModelType.AUTOMODEL_VISION2SEQ,
    )
)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options
        )
    }
)

print("Qwen 7B 모델 다운로드 및 변환 시작 (시간 소요됨)...")
result = doc_converter.convert("doughnut.pdf")
print(result.document.export_to_markdown())