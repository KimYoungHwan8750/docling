
import logging
from collections.abc import Iterable
from pathlib import Path
import time
from typing import Any

from docling_core.types.doc import (
    DoclingDocument,
    NodeItem,
    PictureClassificationClass,
    PictureClassificationData,
    PictureItem,
)

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, PictureDescriptionApiOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.models.base_model import BaseEnrichmentModel
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling_core.types.doc.document import DescriptionAnnotation


# 이미지를 만났을 때 실행할 파이프 라인
class PictureClassifierPipelineOptions(PdfPipelineOptions):
    do_picture_classifer: bool = True
    do_ocr: bool = True
    generate_picture_images: bool = True
    enable_remote_services: bool = True
    images_scale: float = 2.0
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


class PictureClassifierEnrichmentModel(BaseEnrichmentModel):
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def is_processable(self, doc: DoclingDocument, element: NodeItem) -> bool:
        return self.enabled and isinstance(element, PictureItem)

    def __call__(
        self, doc: DoclingDocument, element_batch: Iterable[NodeItem]
    ) -> Iterable[Any]:
        if not self.enabled:
            return

        for element in element_batch:
            assert isinstance(element, PictureItem)

            # uncomment this to interactively visualize the image
            # element.get_image(doc).show()  # may block; avoid in headless runs
            element.annotations.append(
                PictureClassificationData(
                    provenance="example_classifier-0.0.1",
                    predicted_classes=[
                        PictureClassificationClass(class_name="dummy", confidence=0.42)
                    ],
                )
            )

            

            yield element


class PictureClassifierPipeline(StandardPdfPipeline):
    def __init__(self, pipeline_options: PictureClassifierPipelineOptions):
        super().__init__(pipeline_options)
        self.pipeline_options: PictureClassifierPipeline

        self.enrichment_pipe.append(
            PictureClassifierEnrichmentModel(
                enabled=pipeline_options.do_picture_classifer
            )
        )

    @classmethod
    def get_default_options(cls) -> PictureClassifierPipelineOptions:
        return PictureClassifierPipelineOptions()


def main():
    logging.basicConfig(level=logging.INFO)


    pipeline_options = PictureClassifierPipelineOptions()
    pipeline_options.images_scale = 2.0
    pipeline_options.generate_picture_images = True

    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=PictureClassifierPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )
    result = doc_converter.convert("/home/kyh/docling/intp_electronic.pdf")

    for element, _level in result.document.iterate_items():
        if isinstance(element, PictureItem):
            for annotation in element.annotations:
                if isinstance(annotation, DescriptionAnnotation):
                    print(f"Description: {annotation}")

if __name__ == "__main__":
    main()