FROM python:3.10.19

WORKDIR /app

COPY server/requirements.txt .
RUN pip install -r requirements.txt
COPY server/ .
COPY protos/bge_embed.proto ./protos/bge_embed.proto
COPY protos/bge_rerank.proto ./protos/bge_rerank.proto
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. protos/bge_embed.proto
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. protos/bge_rerank.proto

CMD ["python", "wnp_rag_llm.py"]