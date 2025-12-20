import logging
import os
from pathlib import Path

from docling_core.types.doc.document import TableItem
import requests
from docling_core.types.doc import PictureItem, TextItem
from dotenv import load_dotenv

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    PictureDescriptionApiOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

### Example of PictureDescriptionApiOptions definitions

#### Using vLLM
# Models can be launched via:
# $ vllm serve MODEL_NAME


def vllm_local_options(model: str):
    options = PictureDescriptionApiOptions(
        url="http://192.168.0.99:8000/v1/chat/completions",
        params=dict(
            model=model,
            seed=42,
            max_completion_tokens=250,
        ),
        prompt="이미지에 대한 간단한 설명, 차트나 도표가 포함될 경우 누락되는 정보 없도록 마크다운으로 변환",
        timeout=90,
    )
    return options
    
def main():
    logging.basicConfig(level=logging.INFO)

    input_doc_path = "/home/kyh/docling/intp_electronic.pdf"

    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        enable_remote_services=True,  # <-- this is required!
        generate_picture_images=True,
        images_scale=2.0
    )
    pipeline_options.do_picture_description = True
    pipeline_options.picture_description_options = vllm_local_options(model="My_Model")

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
    result = doc_converter.convert(input_doc_path)

    for element, _level in result.document.iterate_items():
        if isinstance(element, PictureItem):
            print(
                f"Picture {element.self_ref}\n"
                f"Caption: {element.captions}\n"
            )
            for annotation in element.annotations:
                print(f"Annotation: {annotation}")
            print(f"Context: {element.export_to_markdown(doc=result.document)}")
        if isinstance(element, TextItem):
            print(f"Text: {element.text}")
        if isinstance(element, TableItem):
            print(f"Table: {element.data}")


if __name__ == "__main__":
    main()