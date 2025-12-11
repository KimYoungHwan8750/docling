from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import VlmPipelineOptions
from docling.datamodel import vlm_model_specs
from docling.datamodel.pipeline_options import AcceleratorOptions, AcceleratorDevice

# ✅ SmolDocling으로 변경 (가장 작음)
pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.SMOLDOCLING_TRANSFORMERS,
    accelerator_options=AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.CUDA
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

print("SmolDocling (최소 VLM) 모델로 변환 시작.")
result = doc_converter.convert("eng_payment_method.pdf")
print(result.document.export_to_markdown())