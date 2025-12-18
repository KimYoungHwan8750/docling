import logging
import os
from pathlib import Path

import requests
from docling_core.types.doc import PictureItem
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
            max_completion_tokens=200,
        ),
        prompt="이 이미지를 설명하고 한국의 기업 삼성에 대해 설명해",
        timeout=90,
    )
    return options
    
def main():
    logging.basicConfig(level=logging.INFO)

    input_doc_path = "doughnut.pdf"

    pipeline_options = PdfPipelineOptions(
        enable_remote_services=True  # <-- this is required!
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
                f"Caption: {element.caption_text(doc=result.document)}\n"
                f"Annotations: {element.annotations}"
            )


if __name__ == "__main__":
    main()